import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
from tqdm import tqdm 
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from torch_geometric.nn import global_mean_pool
from rl_env import MultiSurveyEnv
import joblib 

#

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, feature_dim, hidden_dim, n_actions):
        super(DQN, self).__init__()
        self.layer1 = GCNConv(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)
        self.edge_embedding = nn.Sequential(
            nn.Linear(3,1), 
            nn.ReLU(),
            nn.Linear(1, 1), 
            nn.Sigmoid()
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, data, latent_inference=False, return_latent=False):
        if latent_inference:
            return self.layer3(data)
        else:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            edge_weight = self.edge_embedding(torch.cat([edge_index, edge_weight.T], dim=0).T)
            x = F.relu(self.layer1(x, edge_index, edge_weight=edge_weight))
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.layer2(x))
        if return_latent:
            return x
        return self.layer3(x)

 

class SurveyVehicle(): 
    def __init__(self, feature_dim: int, 
                 hidden_dim: int, 
                 n_actions: int,
                 device: torch.device,
                 policy_net_path: str, 
                 target_net_path: str): 
        self.policy_net = DQN(feature_dim, hidden_dim, n_actions).to(device)
        self.target_net = DQN(feature_dim, hidden_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.load_state_dict(torch.load(policy_net_path, map_location=device))
        self.target_net.load_state_dict(torch.load(target_net_path, map_location=device))

    def set_eval(self,): 
        self.policy_net.eval()
        self.target_net.eval()
    
    def get_latent_code(self, data): 
        return self.policy_net(data, return_latent=True)

class InspectionVehicle():
    def __init__(self, feature_dim: int, 
                 hidden_dim: int, 
                 n_actions: int,
                 device: torch.device,
                 policy_net_path: str, 
                 target_net_path: str): 
        self.policy_net = DQN(feature_dim, hidden_dim, n_actions).to(device)
        self.target_net = DQN(feature_dim, hidden_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.load_state_dict(torch.load(policy_net_path, map_location=device))
        self.target_net.load_state_dict(torch.load(target_net_path, map_location=device))
    
    def set_eval(self,):
        self.policy_net.eval()
        self.target_net.eval()
    
    def select_action_from_latent_code(self, latent_code): 
        return self.policy_net(latent_code, latent_inference=True).max(1)[1].view(1, 1)

    def select_action_from_built_graph(self, data): 
        return self.policy_net(data, latent_inference=False).max(1)[1].view(1, 1)



def select_action(state, eps_greed=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or not eps_greed:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.choice(n_actions)]], device=device, dtype=torch.long) #epsilon greedy, just choose a random action


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = Batch.from_data_list([s for s in batch.next_state
                                                if s is not None])
    state_batch = Batch.from_data_list(list(batch.state))
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

def run_eval_loop(num_surveys, num_sites, rand=False, rand2=False, exh=False):
    episode_rewards = []
    all_preds = []
    all_gts = []
    ep_durations = []
    mean_acc = []
    mean_recall = []
    # past_traj = np.load('/home/advaith/Documents/optix_sidescan/trajectory_views_IMVP14.npy', allow_pickle=True)
    survey_vehicle.set_eval()
    inspection_vehicle.set_eval()
    traj_cnt = 0
    for i_survey in tqdm(range(num_surveys), desc='Survey'):
        r = 0
        survey_preds = []
        survey_gts = []
        site_idx = np.random.choice(len(env.val_dataset), num_sites, replace=False)
        for site_id in site_idx:
            start_angle = np.random.choice(num_angles)
            # Initialize the environment and get its state
            env.update_target_id(site_id)
            state = env.get_init_state(start_angle, compressor=compressor_model)
            # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state = state.to(device)
            #run the episode
            # past_traj_len = len(past_traj[traj_cnt])
            for t in count():
                #the first pass of the object must happen using the survey vehicle: 
                action = inspection_vehicle.select_action_from_built_graph(state)

                if exh:
                    action = torch.tensor([[t % n_actions]], device=device, dtype=torch.long)

                observation, reward, pred = env.step(action.item(), compressor=compressor_model)

                r += reward
                reward = torch.tensor([reward], device=device)
                done = observation is None

                if not done:
                    next_state = observation
                    # next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_state = next_state.clone().to(device)
                else:
                    next_state = None
                    #after the episode is done, we evaluate the prediction using the reward

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                # optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                # target_net_state_dict = target_net.state_dict()
                # policy_net_state_dict = policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                # target_net.load_state_dict(target_net_state_dict)

                if done:
                    ep_durations.append(env.state)
                    if env.state == []:
                        print("oi")
                    survey_gts.append(env.lbl.item())
                    survey_preds.append(pred)
                    all_preds.append(pred)
                    all_gts.append(env.lbl.item())
                    # plot_durations()
                    break
            traj_cnt += 1
        episode_rewards.append(r/len(env.val_dataset))
        #save the weights for the networks 
        # if i_episode % 100 == 0: 
        #     torch.save(policy_net.state_dict(), 'rl_weights/policy_net_{}.pth'.format(i_episode))
        #     torch.save(target_net.state_dict(), 'rl_weights/target_net_{}.pth'.format(i_episode))
        survey_preds = np.array(survey_preds)
        survey_gts = np.array(survey_gts)
        acc = (survey_preds == survey_gts).mean()
        mean_acc.append(acc)

        #calculate the recall using sklearn 
        from sklearn.metrics import recall_score
        recall = recall_score(survey_gts, survey_preds, average='macro')
        mean_recall.append(recall)

    #save ep_durations as numpy array 
    ep_durations = np.array(ep_durations, dtype=object)
    np.save('VIEW_TRAJ_MY_MODEL_proper{}_sites.npy'.format(num_sites), ep_durations)
    np.save('acc_proper.npy', mean_acc)
    np.save('recall_proper.npy', mean_recall)
    #calculate accuracy 
    acc = np.mean(mean_acc)
    recall = np.mean(mean_recall)
    return {"accuracy": acc, "recall": recall}

    # looks_vs_acc = []
    # ls = np.array([len(x) for x in ep_durations])
    # for num_looks in range(1, num_angles):
    #     valid_inds = np.where(ls == num_looks)[0]
    #     valid_preds = np.array(all_preds)[valid_inds]
    #     valid_gts = np.array(all_gts)[valid_inds]
    #     looks_vs_acc.append((valid_preds == valid_gts).mean())
    
    #plot the looks vs acc
    # plt.plot(np.arange(1, num_angles), looks_vs_acc)
    # plt.xlabel('Number of looks')
    # plt.ylabel('Accuracy')
    # plt.savefig('looks_vs_acc.png')

    
    # print('Recall: {}'.format(recall))
    # print('Accuracy: {}'.format(acc))
    # print('Complete')
    # plt.plot(episode_rewards)
    # plt.ioff()
    # plt.show()
    # plt.savefig("Rewards")


def run_eval_loop_sim_si(num_surveys, num_sites, rand=False, rand2=False, exh=False):
    episode_rewards = []
    all_preds = []
    all_gts = []
    ep_durations = []
    mean_acc = []
    target_net.eval()
    policy_net.eval()
    mean_recall = []
    # past_traj = np.load('/home/advaith/Dcuments/optix_sidescan/trajectory_views_IMVP14.npy', allow_pickle=True)
    traj_cnt = 0
    for i_survey in tqdm(range(num_surveys), desc='Survey'):
        r = 0
        survey_preds = []
        survey_gts = []
        site_idx = np.random.choice(len(env.val_dataset), num_sites, replace=False)
        for site_id in site_idx:
            start_angle = np.random.choice(num_angles)
            # Initialize the environment and get its state
            env.update_target_id(site_id)
            state = env.get_init_state(start_angle)
            # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state = state.to(device)
            #run the episode
            # past_traj_len = len(past_traj[traj_cnt])
            for t in count():
                action = select_action(state, eps_greed=False)
                if rand: 
                    action = torch.tensor([[np.random.choice(n_actions)]], device=device, dtype=torch.long)
                if rand2: 
                    if t > past_traj_len-2:
                        action = torch.tensor([6]) #terminate
                    else:
                        #randomly choose action not terminate
                        action = torch.tensor([[np.random.choice(n_actions-1)]], device=device, dtype=torch.long)
                if exh:
                    action = torch.tensor([[t % n_actions]], device=device, dtype=torch.long)
                observation, reward, pred = env.step(action.item())
                r += reward
                reward = torch.tensor([reward], device=device)
                done = observation is None

                if not done:
                    next_state = observation
                    # next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_state = next_state.clone().to(device)
                else:
                    next_state = None
                    #after the episode is done, we evaluate the prediction using the reward

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                # optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                # target_net_state_dict = target_net.state_dict()
                # policy_net_state_dict = policy_net.state_dict()
                # for key in policy_net_state_dict:
                #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                # target_net.load_state_dict(target_net_state_dict)

                if done:
                    ep_durations.append(env.state)
                    if env.state == []:
                        print("oi")
                    survey_gts.append(env.lbl.item())
                    survey_preds.append(pred)
                    all_preds.append(pred)
                    all_gts.append(env.lbl.item())
                    # plot_durations()
                    break
            traj_cnt += 1
        episode_rewards.append(r/len(env.val_dataset))
        #save the weights for the networks 
        # if i_episode % 100 == 0: 
        #     torch.save(policy_net.state_dict(), 'rl_weights/policy_net_{}.pth'.format(i_episode))
        #     torch.save(target_net.state_dict(), 'rl_weights/target_net_{}.pth'.format(i_episode))
        survey_preds = np.array(survey_preds)
        survey_gts = np.array(survey_gts)
        acc = (survey_preds == survey_gts).mean()
        mean_acc.append(acc)

        #calculate the recall using sklearn 
        from sklearn.metrics import recall_score
        recall = recall_score(survey_gts, survey_preds, average='macro')
        mean_recall.append(recall)

    #save ep_durations as numpy array 
    ep_durations = np.array(ep_durations, dtype=object)
    np.save('VIEW_TRAJ_MY_MODEL_proper{}_sites.npy'.format(num_sites), ep_durations)
    np.save('acc_proper.npy', mean_acc)
    np.save('recall_proper.npy', mean_recall)
    #calculate accuracy 
    acc = np.mean(mean_acc)
    recall = np.mean(mean_recall)

    looks_vs_acc = []
    ls = np.array([len(x) for x in ep_durations])
    for num_looks in range(1, num_angles):
        valid_inds = np.where(ls == num_looks)[0]
        valid_preds = np.array(all_preds)[valid_inds]
        valid_gts = np.array(all_gts)[valid_inds]
        looks_vs_acc.append((valid_preds == valid_gts).mean())
    
    #plot the looks vs acc
    plt.plot(np.arange(1, num_angles), looks_vs_acc)
    plt.xlabel('Number of looks')
    plt.ylabel('Accuracy')
    plt.savefig('looks_vs_acc.png')

    
    print('Recall: {}'.format(recall))
    print('Accuracy: {}'.format(acc))
    print('Complete')
    plt.plot(episode_rewards)
    plt.ioff()
    plt.show()
    plt.savefig("Rewards")

def run_train_loop():
    episode_rewards = []
    #sanity check 
    i_episode = 0
    torch.save(policy_net.state_dict(), 'rl_weights_{}_no_avg/policy_net_{}.pth'.format(seed, i_episode))
    torch.save(target_net.state_dict(), 'rl_weights_{}_no_avg/target_net_{}.pth'.format(seed, i_episode))
    for i_episode in tqdm(range(num_episodes), desc='Episode'):
        rs = []
        ep_durations = []
        for target_id in tqdm(range(len(env.val_dataset)),desc='dataset'):
            start_angle = np.random.choice(num_angles)
            # Initialize the environment and get its state
            env.update_target_id(target_id)
            state = env.get_init_state(start_angle)
            state = state.to(device)
            r = 0
            for t in count():
                action = select_action(state)
                observation, reward, pred = env.step(action.item())
                r += reward
                reward = torch.tensor([reward], device=device)
                done = observation is None

                if not done:
                    next_state = observation
                    # next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    next_state = next_state.clone().to(device)
                else:
                    next_state = None

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    rs.append(r)
                    # plot_durations()
                    break

        episode_rewards.append(np.mean(rs))
        print("Episode {} reward: {}".format(i_episode, np.mean(rs)))
        #save the weights for the networks 
        if i_episode % 10 == 0: 
            torch.save(policy_net.state_dict(), 'rl_weights_{}_no_avg/policy_net_{}.pth'.format(seed, i_episode))
            torch.save(target_net.state_dict(), 'rl_weights_{}_no_avg/target_net_{}.pth'.format(seed, i_episode))
    print('Complete')
    plt.plot(episode_rewards)
    plt.ioff()
    plt.show() 
    plt.savefig("Rewards.png")
 
# run_train_loop()
    # policy_net = DQN(1000, 128, n_actions).to(device)
# target_net = DQN(1000, 128, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# policy_net.load_state_dict(torch.load('/mnt/syn/advaiths/optix_sidescan/rl_weights_14_proper/policy_net_280.pth', map_location=device))
# target_net.load_state_dict(torch.load('/mnt/syn/advaiths/optix_sidescan/rl_weights_14_proper/target_net_280.pth', map_location=device))
    # seed = 14
seeds = [14, 1917, 8125]
acc = []
recall = []
for seed in seeds:
    # seed = 1917
    # seed = 8125
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("SEED", seed)
    # env = gym.make("CartPole-v1")
    weights_path = '/mnt/syn/advaiths/DoD SAFE-2twjN5BZ5TQnffsY/version_9/best-epoch=19-val_accuracy=0.98.ckpt'
    print("LOADED FROM: ", weights_path)
    num_angles = 6
    data_dir = '/mnt/syn/advaiths/NRL_GMVATR/harder_scenes2'
    big_rewards = []
    split = 'test'
    env = MultiSurveyEnv(weights_path, num_angles, data_dir, seed, 0, gnn_layers=2, gnn_hidden_dim=64, p=0.5, split=split)

    compressor_model = joblib.load('/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/compressor_weights/pca_18.pkl')

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    # seed = 14
    # Get number of actions from gym action space
    n_actions = 7
    # Get the number of state observations
    state = env.get_init_state(0) 

    #there are two types of vehicles 

    survey_vehicle = SurveyVehicle(feature_dim=1000, 
                                hidden_dim=128, 
                                n_actions=n_actions,
                                device=device,
                                policy_net_path='/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/rl_weights_14_proper/policy_net_280.pth',
                                target_net_path='/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/rl_weights_14_proper/target_net_280.pth')

    inspection_vehicle = InspectionVehicle(feature_dim=1000,
                                            hidden_dim=128,
                                            n_actions=n_actions,
                                            device=device,
                                            policy_net_path='/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/rl_weights_14_proper/policy_net_280.pth',
                                            target_net_path='/mnt/syn/advaiths/NRL_GMVATR/optix_sidescan/rl_weights_14_proper/target_net_280.pth')

    # optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0

    metrics = run_eval_loop(100, 16, exh=False, rand=False, rand2=False)
    acc.append(metrics['accuracy'])
    recall.append(metrics['recall'])

print("ACCURACY: ", np.mean(acc))
print("RECALL: ", np.mean(recall))

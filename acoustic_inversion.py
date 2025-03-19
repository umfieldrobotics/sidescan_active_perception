### Writing my own raytracer 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import torch
from tqdm import tqdm
from IPython.display import HTML, display, Image
from matplotlib import animation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SIMULATION_STEPS = 2.0  # meters
SIMULATION_RANGE = torch.arange(0, 100 + SIMULATION_STEPS, SIMULATION_STEPS)
TRANSMISSION_ANGLE_RANGE = torch.deg2rad(torch.arange(-10, 11, 1)).to(device)  # angle of transmission in rad

#write the forward model that takes in theta parameters and integrates rays 
def forward_model(params): 
    SOUND_SPEED_MIN = params[0]
    SOUND_SPEED_MAX = params[1]

    DEPTH_MIN = 0  # Ocean Surface
    DEPTH_MAX = 30  # meters
    DEPTH_MIN_SPEED = 10 # depth where minimum speed of sound is observed
    DEPTH_RANGE = torch.arange(DEPTH_MIN, DEPTH_MAX + 1).to(device)

    SOUND_GRADIENT_SHALLOW = (SOUND_SPEED_MIN - SOUND_SPEED_MAX) / (DEPTH_MIN_SPEED - DEPTH_MIN)
    SOUND_GRADIENT_DEEP = (SOUND_SPEED_MAX - SOUND_SPEED_MIN) / (DEPTH_MAX - DEPTH_MIN_SPEED)

    SOURCE_DEPTH = DEPTH_MIN_SPEED
    SOURCE_SOUND_SPEED = SOUND_SPEED_MIN

    
    angle_0_ind = torch.argwhere(TRANSMISSION_ANGLE_RANGE == 0)[0][0]

    # Instantiate our matrices
    R = torch.zeros(len(TRANSMISSION_ANGLE_RANGE)).to(device)
    z = torch.zeros_like(R).to(device)
    c = torch.zeros_like(R).to(device)
    theta = torch.zeros_like(R).to(device)

    z[:] = SOURCE_DEPTH  # we put the source at the depth of min sound speed
    R[:] = -SOURCE_SOUND_SPEED / torch.cat((
        SOUND_GRADIENT_SHALLOW * torch.cos(TRANSMISSION_ANGLE_RANGE[:angle_0_ind+1]),
        SOUND_GRADIENT_DEEP * torch.cos(TRANSMISSION_ANGLE_RANGE[angle_0_ind+1:])))
    c[:] = SOUND_SPEED_MIN
    theta[:] = TRANSMISSION_ANGLE_RANGE

    wave_guides = [z]
    for j in range(1, len(SIMULATION_RANGE)):

        case_1_mask = (z == SOURCE_DEPTH) & (theta == 0)
        case_2_mask = (z < SOURCE_DEPTH) | (z == SOURCE_DEPTH) & (theta > 0)
        case_3_mask = (z > SOURCE_DEPTH) | (z == SOURCE_DEPTH) & (theta < 0)

        R_new = R*case_1_mask + (-c / (SOUND_GRADIENT_SHALLOW * torch.cos(theta)))*case_2_mask + (-c / (SOUND_GRADIENT_DEEP * torch.cos(theta)))*case_3_mask
        theta_new = 0*case_1_mask + (case_2_mask | case_3_mask)*torch.arcsin(SIMULATION_STEPS / R + torch.sin(theta))
        dz = (case_2_mask | case_3_mask)* R * (torch.cos(theta) - torch.cos(theta_new))
        z_new = case_1_mask*SOURCE_DEPTH + (case_2_mask | case_3_mask)*(z + dz)        
        c_new = case_1_mask*SOURCE_SOUND_SPEED + case_2_mask*(SOURCE_SOUND_SPEED + SOUND_GRADIENT_SHALLOW * (z_new - SOURCE_DEPTH)) + (SOURCE_SOUND_SPEED + SOUND_GRADIENT_DEEP * (z_new - SOURCE_DEPTH))*case_3_mask

        # for i in range(0, len(TRANSMISSION_ANGLE_RANGE)):
        #     if (z[i] == SOURCE_DEPTH) and (theta[i] == 0):
        #         c[i] = SOURCE_SOUND_SPEED
        #         theta[i] = 0
        #         dz[i] = 0
        #         z[i] = SOURCE_DEPTH
                
        #     elif (z < SOURCE_DEPTH) or (z == SOURCE_DEPTH and theta > 0):
        #         R_new = -c / (SOUND_GRADIENT_SHALLOW * torch.cos(theta))
        #         theta_new = torch.arcsin(
        #             SIMULATION_STEPS / R + torch.sin(theta)
        #         )
        #         dz = R * (torch.cos(theta) - torch.cos(theta_new))
        #         z_new = z + dz
        #         c_new = SOURCE_SOUND_SPEED + SOUND_GRADIENT_SHALLOW * (z_new - SOURCE_DEPTH)
                
        #     elif (z[i, j-1] > SOURCE_DEPTH) or (z[i, j-1] == SOURCE_DEPTH and theta[i, j-1] < 0):
        #         R_new = -c / (SOUND_GRADIENT_DEEP * torch.cos(theta))
        #         theta_new = torch.arcsin(
        #             SIMULATION_STEPS / R + torch.sin(theta)
        #         )
        #         dz = R * (torch.cos(theta) - torch.cos(theta_new))
        #         z_new = z + dz
        #         c_new = SOURCE_SOUND_SPEED + SOUND_GRADIENT_DEEP * (z_new - SOURCE_DEPTH)
            
        z = z_new
        c = c_new
        R = R_new
        theta = theta_new
        wave_guides.append(z)

        
    return torch.stack(wave_guides).T

gt = forward_model([1430, 1550])
hloss = torch.nn.HuberLoss()
# torch.autograd.set_detect_anomaly(True)

num_iters = 100000 
lr = 1e-2
save = True
#uniformly between range of 1400 and 1600
masking_percent = 0.50
rand_mask = torch.rand_like(gt) > masking_percent
params = torch.tensor([1300.0, 1700.0], requires_grad=True, device=device)
optim = torch.optim.Adam([params], lr=lr)
unfreeze_schedule = 100 #every 500 iters we unfreeze another range segment
initial_masking = torch.zeros_like(gt).bool()
initial_masking[:, 0:5] = 1

#open a 3d plot 
angles = torch.tensor([0.0, 60.0, 220.0])
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(0, 30)

#plot the gt points rotated by angle in angles 
for angle in angles:
    x = SIMULATION_RANGE.expand_as(gt)[rand_mask] * torch.cos(torch.deg2rad(angle))
    y = SIMULATION_RANGE.expand_as(gt)[rand_mask] * torch.sin(torch.deg2rad(angle))
    z = gt[rand_mask]
    ax.plot(x.cpu(), y.cpu(), z.cpu(), label="gt")

plt.title("3D Visualization of Raytraced Beams between AUVs")
plt.show()

plt.figure(figsize=(12,8))
plt.scatter((SIMULATION_RANGE.expand_as(gt))[rand_mask], gt[rand_mask].cpu()) 
plt.plot(SIMULATION_RANGE, gt.cpu().transpose(1,0), 'g') 

plt.xlim(0, 100)
plt.ylim(0, 30)
plt.gca().invert_yaxis()
plt.title('Acoustic Ray Tracing')
plt.xlabel('Range (m)')
plt.ylabel('Depth (m)')
plt.savefig("gt.png")

for i in tqdm(range(num_iters)):
    optim.zero_grad()
    W_hat = forward_model(params)
    # if i % unfreeze_schedule == 0:
    #     initial_masking[:, i // unfreeze_schedule] = 1
    #     plt.figure(figsize=(12,8))
    #     plt.scatter((SIMULATION_RANGE.expand_as(gt))[rand_mask], gt[rand_mask].cpu()) 
    #     plt.plot(SIMULATION_RANGE, gt.cpu().transpose(1,0), 'g') 

    #     plt.xlim(0, 100)
    #     plt.ylim(0, 30)
    #     plt.gca().invert_yaxis()
    #     plt.title('Acoustic Ray Tracing')
    #     plt.xlabel('Range (m)')
    #     plt.ylabel('Depth (m)')
    #     plt.savefig("gt_{}.png".format(i))
        
    
    loss = hloss(W_hat[rand_mask], gt[rand_mask])
    loss.backward()
    optim.step()
    if i % 10 == 0:
        #log the loss in tqdm 
        tqdm.write(f'Loss: {loss.item()}')
    # if i % 50 == 0 and save: 
    #     plt.figure(figsize=(12,8))
    #     #plot all of gt at once in one color 
    #     plt.plot(SIMULATION_RANGE, gt.cpu().transpose(0,1), 'g') 
    #     plt.plot(SIMULATION_RANGE,  W_hat.detach().cpu().transpose(0,1), 'r')
    #     plt.gca().invert_yaxis()
    #     plt.title('Acoustic Ray Tracing')
    #     plt.xlim(0, 50)
    #     plt.ylim(6, 16)
    #     plt.xlabel('Range (m)')
    #     plt.ylabel('Depth (m)')
    #     plt.legend()
    #     #save the image of the plot  with 4 digits 
    #     plt.savefig(f'imgs/iter_{i:04d}.png')
    #     # plt.show()

np.save("gt", gt.cpu())

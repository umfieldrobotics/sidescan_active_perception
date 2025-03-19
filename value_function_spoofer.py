import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dubins import *
from matplotlib import cm
# from python_tsp.heuristics import solve_tsp_local_search
seed = 14
np.random.seed(seed)
def plot_polygon_with_values(vertices, triangle_values):
    # Ensure the polygon is closed (first and last vertex are the same)
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])

    # Calculate the center of the polygon
    center = np.mean(vertices, axis=0)

    # Create the plot and plot the polygon
    plt.plot(vertices[:, 0], vertices[:, 1], 'b-')

    # Normalize the triangle_values to range [0, 1]
    norm = plt.Normalize(min(triangle_values), max(triangle_values))
    cmap = cm.get_cmap('cool')

    # Annotate each triangle with its corresponding value and draw the triangle
    for i in range(len(vertices) - 1):
        v1 = vertices[i]
        v2 = vertices[i + 1]
        triangle_vertices = [v1, v2, center]

        triangle = patches.Polygon(triangle_vertices, closed=True, edgecolor='g', facecolor=cmap(norm(triangle_values[i])))
        plt.gca().add_patch(triangle)
        plt.annotate(f"{triangle_values[i]:.2f}", xy=(v1 + v2 + center) / 3, color='b', ha='center', va='center')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Map with View Values')
    plt.grid(True)
    
    #axis equal 

def create_random_value_function_one_point(poly_order):
    values = np.zeros((poly_order))
    
    #populate values with random integers 
    values = np.random.randint(low=0, high=100, size=(poly_order))
    return values

# from python_tsp.heuristics import solve_tsp_local_search
from matplotlib import pyplot as plt
import numpy as np 
#generate waypoints in a lawnmower pattern within a fixed grid 
def generate_waypoints(size, vertical_pts, horizontal_spacing):
    nv = vertical_pts 
    nh = size[1]//horizontal_spacing
    pts = []
    for i in range(nh+1): 
        y = np.linspace(0, size[0], nv)
        x = np.ones(nv)*i*horizontal_spacing
        if i % 2 == 0:
            #reverse the points
            x = x[::-1]
            y = y[::-1]
        pts.append(np.vstack((x,y)).T)

    return np.vstack(pts)
    
num_targets = 16
grid_size = (1200, 1200)
num_vert = 25
v_step = grid_size[0]/(num_vert-1)
hor_spacing = 60

num_iters = 100
survey_alt = 5
reack_alt = 3
sensor_range = 30
nss = [num_targets]
poly_order = 6
viewing_radius = 12
pass_len = 75
space = 100
dists = []
for ni in range(num_iters):
    contacts = np.random.uniform(0, grid_size[0], (num_targets, 2))
    
    #i am travelling now 
    full_path = []
   

print("Average inter dist ", np.mean(dists))
print("Std distance travelled: ", np.std(dists))
print("Intra target dist: ", num_targets*pass_len)
print("total dist", np.mean(dists) + num_targets*pass_len)
plt.hist(dists)
plt.savefig("hist_dist.png")
#calculate a cost matrix for the pts 
# cost_matrix = np.zeros((len(pts), len(pts)))
# for i in range(len(pts)):
#     for j in range(len(pts)):
#         cost_matrix[i][j] = np.linalg.norm(pts[i] - pts[j])
# #solve the tsp
# cost_matrix[:,0] = 0
# path, distance = solve_tsp_local_search(cost_matrix)
# #plot the path
# plt.figure(figsize=(10,10))
# plt.plot(pts[path][:,0], pts[path][:,1], 'r--')
# #draw an x at the start 
# plt.scatter(pts[path[0]][0], pts[path[0]][1], marker='x', s=100)
# plt.figure(figsize=(10,10))
# plt.scatter(np.array(pts)[:,0], np.array(pts)[:,1])
# plt.scatter(contacts[:,0], contacts[:,1], marker='x', color='g', s=100)
# #draw arrows between consecutive points 
# for i in range(len(full_path)-1):
#     plt.arrow(full_path[i][0], full_path[i][1], full_path[i+1][0]-full_path[i][0], full_path[i+1][1]-full_path[i][1], width=0.3, color='r', length_includes_head=True)
# #calculate total distance travelled along path 

# print(dist)

#PARAMETERS



dist_inter = []
dist_intra = []
pts = generate_waypoints(grid_size, num_vert, hor_spacing)
accuracies = np.load('/mnt/syn/advaiths/optix_sidescan/proper_acc.npy')
recalls = np.load('/mnt/syn/advaiths/optix_sidescan/proper_recall.npy')


chosen_nodes = np.load('/mnt/syn/advaiths/optix_sidescan/VIEW_TRAJ_MY_MODEL_proper16_sites.npy', allow_pickle=True)
# chosen_nodes = [np.random.choice(np.arange(poly_order), size=(2)) for i in range(nss[0]*num_iters)]
# chosen_nodes = [np.arange(poly_order) for i in range(nss[0]*num_iters)]
# chosen_nodes = [[0] for i in range(nss[0]*num_iters)]
# chosen_nodes = [[] for i in range(nss[0]*num_iters)]
from tqdm import tqdm 
for ns in nss:
    for k in tqdm(range(num_iters)):
        full_path = []
        contacts = np.vstack((np.random.uniform(0, grid_size[1], (ns, )), np.random.uniform(0, grid_size[0], (ns, )))).T
        # map_bounds = [0, 0, 100, 100]
        num_sites = ns
        
        my_slice_of_views = chosen_nodes[k*num_sites:(k+1)*num_sites]
        # for i in range(num_sites):
        #     my_slice_of_views[i] += i*poly_order
        # my_slice_of_views = np.concatenate(my_slice_of_views)
        # num_nodes = my_slice_of_views.shape[0]
        num_nodes = sum([len(x) for x in my_slice_of_views])
        values = []
        for i in range(num_sites):
            values.append(create_random_value_function_one_point(poly_order))
        values = np.array(values).reshape((num_sites, poly_order)).astype(np.float32)
        #randomly choose num_sites coordinate locations within map bounds that are at least viewing_radius apart
        site_coords = []
        # for i in range(num_sites):
        #     #generate random coordinate
        #     rand_coord = np.random.randint(low=map_bounds[0], high=map_bounds[2], size=(2))
        #     #check if it is at least viewing_radius away from all other sites 
        #     # while len(site_coords) > 0 and np.min(np.linalg.norm(site_coords - rand_coord, axis=-1)) < space:
        #     #     rand_coord = np.random.randint(low=map_bounds[0], high=map_bounds[2], size=(2))
        #     site_coords.append(rand_coord)
        # site_coords = np.array(site_coords)
        site_coords = contacts

        #write a function that creates a hexagon centered at a point with a given radius and side length
        def create_passes_fb(num_sites, centers, radius, side_length, num_sides):
            #output is a numpy array of num_sites * num_sides * 2 (forward/backward) * 2 (entry/exit) * 2 x/y coordinates
            dtheta = 2*np.pi/num_sides
            passes = np.zeros((num_sites, num_sides, 2, 2, 2))
            for i in range(num_sites):
                center = centers[i]
                for j in range(num_sides):
                    theta = j*dtheta

                    entry_point = center + np.array([radius, -side_length/2])
                    exit_point = center + np.array([radius, side_length/2])

                    #rotate both by theta
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    entry_point = np.matmul(R, entry_point - center) + center
                    exit_point = np.matmul(R, exit_point - center) + center
                    passes[i, j, 0, 0] = entry_point #first approach forward
                    passes[i, j, 0, 1] = exit_point
                    passes[i,j, 1, 0] = exit_point #second approach backward
                    passes[i,j, 1, 1] = entry_point

            return passes

        #this is for the single-view baselines that can be solved using only a TSP solver. 
        def create_passes_single_hit(num_sites, centers, radius, side_length, num_sides):
            #output is a numpy array of num_sites * num_sides  * 2 (entry/exit) * 2 x/y coordinates
            dtheta = 2*np.pi/num_sides
            passes = np.zeros((num_sites, num_sides,  2, 2))
            for i in range(num_sites):
                center = centers[i]
                for j in range(num_sides):
                    theta = j*dtheta

                    entry_point = center + np.array([radius, -side_length/2])
                    exit_point = center + np.array([radius, side_length/2])

                    #rotate both by theta
                    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    entry_point = np.matmul(R, entry_point - center) + center
                    exit_point = np.matmul(R, exit_point - center) + center
                    passes[i, j, 0] = entry_point #first approach forward
                    passes[i, j, 1] = exit_point

            return passes



        # mat = create_passes_fb(num_sites, site_coords, viewing_radius, pass_len, poly_order)
        mat = create_passes_single_hit(num_sites, site_coords, viewing_radius, pass_len, poly_order)

        #experimental code 
        # plt.figure()
        # plt.scatter(site_coords[:,0], site_coords[:,1], marker='x', color='g', s=100)
        # col_mat = mat.reshape(-1, 2)
        # path = []
        # for pt in range(col_mat.shape[0]-1):
        #     start_point = col_mat[pt]
        #     end_point = col_mat[pt+1]
            

        #     start_yaw = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        #     if pt == col_mat.shape[0] - 2:
        #         end_yaw = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        #     else:
        #         next_end_point = col_mat[pt+2]
        #         end_yaw = np.arctan2(next_end_point[1] - end_point[1], next_end_point[0] - end_point[0])
        #     #draw arrow 
        #     # plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], color='r', width=1)

        #     path_x, path_y, path_yaw, mode, _ = plan_dubins_path(start_point[0], start_point[1], start_yaw, end_point[0], end_point[1], end_yaw, 1.0)
            
        #     path.append(np.vstack((path_x, path_y)).T)
        # path = np.concatenate(path, axis=0)
        # plt.plot(path[:,0], path[:,1], 'b--')
        # plt.axis('equal')
        # plt.show()

        mat_for_cost = mat.reshape((-1, 2, 2))

        for i in range(pts.shape[0]-1): 
            current_location = pts[i]
            db_start = current_location 
            db_end = pts[i+1]
            start_yaw = np.arctan2(db_end[1] - db_start[1], db_end[0] - db_start[0])
            if i == pts.shape[0] - 2:
                end_yaw = np.arctan2(db_end[1] - db_start[1], db_end[0] - db_start[0])
            else:
                next_end_point = pts[i+2]
                end_yaw = np.arctan2(next_end_point[1] - db_end[1], next_end_point[0] - db_end[0])
            path_x, path_y, path_yaw, mode, _ = plan_dubins_path(db_start[0], db_start[1], start_yaw, db_end[0], db_end[1], end_yaw, 1.0/4.0)
            full_path.append(np.vstack((path_x, path_y)).T)
            captured = (np.abs(site_coords[:,0] - current_location[0]) < sensor_range) & (np.abs(site_coords[:,1] - current_location[1]) < v_step/2)
            if captured.any():
                # print("Captured")
                # print(contacts[captured])
                survey_pts = []
                for j in np.nonzero(captured)[0]:
                    points_to_visit = mat[j, my_slice_of_views[j]].reshape((-1, 2))
                    valid_entries = points_to_visit[::2]
                    #find the index of gthe closest valid_entry to curr pos
                    closest_idx = 2*np.argmin(np.linalg.norm(valid_entries - current_location, axis=1))
                    #roll the points_to_vist such that the closest_idx is at 0 
                    points_to_visit = np.roll(points_to_visit, -closest_idx, axis=0)
                    survey_pts.append(points_to_visit)
                survey_pts = np.concatenate(survey_pts, axis=0)
                inspection_wp = np.concatenate([current_location[None], survey_pts, current_location[None], db_end[None]], axis=0)

                #calculate cost matrix 
                # cost_matrix = np.zeros((inspection_wp.shape[0], inspection_wp.shape[0]))
                # for i in range(inspection_wp.shape[0]):
                #     for j in range(inspection_wp.shape[0]):
                #         cost_matrix[i,j] = np.linalg.norm(inspection_wp[i] - inspection_wp[j])
                # path, distance = solve_tsp_local_search(cost_matrix)
                # full_path.append(inspection_wp[path])
                # full_path.append(inspection_wp)
                db_path = []
                for pt in range(inspection_wp.shape[0]-2):
                    start_point = inspection_wp[pt]
                    end_point = inspection_wp[pt+1]
                    start_yaw = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
                    next_end_point = inspection_wp[pt+2]
                    end_yaw = np.arctan2(next_end_point[1] - end_point[1], next_end_point[0] - end_point[0])
                    path_x, path_y, path_yaw, mode, _ = plan_dubins_path(start_point[0], start_point[1], start_yaw, end_point[0], end_point[1], end_yaw, 1.0/10.0)
                    
                    db_path.append(np.vstack((path_x, path_y)).T)
                full_path += db_path

        full_path = np.concatenate(full_path, axis=0)
        dist = np.linalg.norm(np.diff(full_path, axis=0), axis=1).sum() + num_nodes*pass_len
        dists.append(dist)
            # mat_for_cost = 
        
        #move at 2.0m/s at traj sample time of 0.1 seconds 
        #create a spline of the entire trajectory and display 


        
        #plot the etrjectory se
        step = 100
        for i in tqdm(range(0, full_path.shape[0]-1, step)):
            subsection = full_path[:i]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(site_coords[:,0], site_coords[:,1], marker='o', color='r', s=100)
            #axis equal 
            ax.axis('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.plot(subsection[:,0], subsection[:,1], 'b--')
            ax.set_xlim([-70, grid_size[0]+70])
            ax.set_ylim([-70, grid_size[1]+70])
            #save figure as image 
            #save formatted as 4 digit number 

            plt.savefig(f"lm/" + format(i//step, '04') + ".png")
        quit()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(full_path[:,0], full_path[:,1], 'b--', linewidth=1)
        ax.scatter(site_coords[:,0], site_coords[:,1], marker='o', color='r', s=100)
        #axis equal 
        ax.axis('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        #add gridlines 
        # ax.grid(True, which='both')
        #make the plot only show the area of interest
        ax.set_xlim([-70, grid_size[0]+70])
        ax.set_ylim([-70, grid_size[1]+70])
        plt.show()

        # ax.set_title("GMVATR + AG + V-QF ASR Trajectory")
        # plt.savefig("ours_traj.png", dpi=600)
        # plt.show()

        #matplotlib plot on blue background 
        # plt.figure(figsize=(10,10))
        # plt.scatter(np.array(pts)[:,0], np.array(pts)[:,1])
        # plt.scatter(site_coords[:,0], site_coords[:,1], marker='x', color='g', s=100)

        # #set xlim and ylim
        # # plt.xlim([0, grid_size[0]])
        # # plt.ylim([0, grid_size[1]])
        # plt.show()
        # #draw arrows between consecutive points 
        # for i in range(len(full_path)-1):
        #     plt.arrow(full_path[i][0], full_path[i][1], full_path[i+1][0]-full_path[i][0], full_path[i+1][1]-full_path[i][1], width=0.3, color='r', length_includes_head=True)
        # plt.show()
        # # pts_forward = mat[:,:,0].reshape((-1, 2)) + np.random.normal(scale=0.001, size=(mat[:,:,0].reshape((-1, 2)).shape))
        # pts_backward = mat[:,:,1].reshape((-1, 2))
        pts_forward = mat.reshape((-1, 2)) 
        # plt.figure()


        # plt.scatter(site_coords[:,0], site_coords[:,1])

        # cost_matrix = np.zeros((num_nodes, num_nodes))
        # # cost_matrix = np.zeros((num_sites*poly_order*2, num_sites*poly_order*2))

        # for i in range(num_nodes): 
        #     for j in range(num_nodes):
        #         if i != j : 
        #             start_idx = my_slice_of_views[i]
        #             end_idx = my_slice_of_views[j]
        #             start_point = mat_for_cost[start_idx,1] #we start at the end of the "from" leg 
        #             end_point = mat_for_cost[end_idx, 0] #we end at the start of the "to" leg
        #             cost_matrix[i, j] = np.linalg.norm(start_point - end_point) #i -> j 
            
        




        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         from_site_cluster = i % poly_order + my_slice_of_views[i % poly_order]
        #         to_site_cluster = j % poly_order + my_slice_of_views[j % poly_order]
        #         start_point = mat_for_cost[i,1] #we start at the end of the "from" leg 
        #         end_point = mat_for_cost[j, 0] #we end at the start of the "to" leg
        #         cost_matrix[i, j] = np.linalg.norm(start_point - end_point) #i -> j 


        #solve the TSP proiblem 
        # permutation, distance = solve_tsp_local_search(cost_matrix)
        # permutation.append(my_slice_of_views[0])

        # #find the maximum distance on the tour between two nodes 
        # dds = []
        # inter_target_dists = []
        # intra_target_dists = []
        # for p in range(len(permutation) - 1):
        #     dds.append(cost_matrix[permutation[p], permutation[p+1]])
        #     if permutation[p] // poly_order != permutation[p+1] // poly_order:
        #         #this means it switched between targets 
        #         inter_target_dists.append(cost_matrix[permutation[p], permutation[p+1]])
        #     else:
        #         intra_target_dists.append(cost_matrix[permutation[p], permutation[p+1]])
            
        # inter_target_dists = np.array(inter_target_dists)
        # intra_target_dists = np.array(intra_target_dists)

        #sort the dds 
        # dds = np.array(dds).reshape(num_sites, poly_order)
        # dds.sort()

        # print("inter target dists: ", np.sum(inter_target_dists))
        # print("intra target dists: ", np.sum(intra_target_dists) + num_sites*poly_order*pass_len)
        # print(f"total distance from doing OID: ", num_sites*poly_order*pass_len)
        # print(f"total distance from doing TSP: ", distance + num_sites*poly_order*pass_len)
        # dist.append(distance + num_sites*poly_order*pass_len)
        # dist_inter.append(np.sum(inter_target_dists))
        # dist_intra.append(np.sum(intra_target_dists) + num_nodes*pass_len)

        # plt.scatter(site_coords[:,0], site_coords[:,1])
        # #t an x in the start location 
        # plt.scatter(mat_for_cost[permutation[0], 1, 0], mat_for_cost[permutation[0], 1, 1], color='k', marker='x')
        # for i in range(len(permutation) - 1):
        #     start_point = mat_for_cost[permutation[i],1] #we start at the end of the "from" leg 
        #     end_point = mat_for_cost[permutation[i+1], 0] #we end at the start of the "to" leg
        #     plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], color='g', width=1, 
        #             linestyle='--')
    
        # for i in range(0, pts_forward.shape[0] - 1, 2):
        #     plt.arrow(pts_forward[i,0], pts_forward[i,1], pts_forward[i+1,0] - pts_forward[i,0], pts_forward[i+1,1] - pts_forward[i,1], color='r', width=1)
        # # plt.scatter(mat_for_cost.reshape(-1,2)[:,0], mat_for_cost.reshape(-1,2)[:,1], color='r')
        # plt.axis('equal')
        # plt.savefig("tour.png")

speed = 2.0
times = np.array(dists)/speed/60/60
nv = [len(cs) for cs in chosen_nodes]
print("Average time", np.mean(times))
print("Std time", np.std(times))

print("mean coverage rate", np.mean(1.44/times))
print("std coverage rate", np.std(1.44/times))

print("mean accuracy", np.mean(accuracies))
print("std accuracy", np.std(accuracies))

print("mean recall", np.mean(recalls))
print("std recall", np.std(recalls))

print("mean CE", np.mean(recalls/times))
print("std CE", np.std(recalls/times))

print("mean AV", np.mean(nv))
print("std AV", np.std(nv))
#read in the en_accuracies.npy
# en_accuracies = np.load('en_accuracies.npy')
# or_accuracies = np.load('OR_accuracies.npy')
# en_CE = np.load('en_CE.npy')
# or_CE = np.load('OR_CE.npy')
# print("inter target dists: {}".format(num_sites), np.mean(dist_inter), np.std(dist_inter))
# print("intra target dists: {}".format(num_sites), np.mean(dist_intra), np.std(dist_intra))
# total_dist_vect = np.array(dist_inter) + np.array(dist_intra)

# print("en_efficiency", (en_CE/total_dist_vect).mean())
# print("or_efficiency", (or_CE/total_dist_vect).mean())
# max_cost = np.max(cost_matrix)
# min_cost = np.min(cost_matrix)
# for i in range(num_sites*poly_order*2):
#     for j in range(num_sites*poly_order*2):
#         start_point = mat_for_cost[i,1] #we start at the end of the "from" leg 
#         end_point = mat_for_cost[j, 0] #we end at the start of the "to" leg
#         #plot the arrrow between the two points and make the color the cost 
#         cost = (cost_matrix[i,j] - min_cost)/(max_cost - min_cost)
#         plt.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1], color=cm.jet(cost), width=0.1, 
#                   linestyle='--')

reward_vector = values.reshape(-1).repeat(2)
for i in range(0, num_sites*poly_order*2, num_sites):
    reward_vector[i:i+poly_order*2] /= reward_vector[i:i+poly_order*2].max()
#draw an arrow between each consecutive pair of points 

# for i in range(0, pts_backward.shape[0] - 1, 2):
#     plt.arrow(pts_backward[i,0], pts_backward[i,1], pts_backward[i+1,0] - pts_backward[i,0], pts_backward[i+1,1] - pts_backward[i,1], color=cm.winter(reward_vector[i]), width=1)
plt.axis('equal')
plt.colorbar()
plt.show()
#display cost matri in plt 
plt.figure()
plt.imshow(cost_matrix)
plt.colorbar()
plt.show()
# #visualize each layer of cost_matrix in subplot 
r = (map_bounds[2] - map_bounds[0])//10
plt.figure()
vertices = r*np.array([[0, 0], [1, 0], [1.5, np.sqrt(3)/2], [1, np.sqrt(3)], [0, np.sqrt(3)], [-0.5, np.sqrt(3)/2]])
for i in range(num_sites):
    plot_polygon_with_values(vertices + site_coords[i], values[i,:])
plt.axis('equal')
plt.colorbar(cm.ScalarMappable(cmap='cool'))
plt.show()

#write the cost matrix to a file
np.savetxt('cost_matrix.txt', cost_matrix)

#write the reward vector to a file
np.savetxt('reward_vector.txt', reward_vector)




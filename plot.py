from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from np2pth import get_system_wrapper, get_controller_wrapper

import importlib
from utils import RK4_system
import time

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
HUGE_SIZE = 25

# Simulation variables
trajectory_type = 'circular'  # 'hover', 'circular', 'figure-8'
plot_type = '3D'  # '2D', '3D', 'time', 'error', 'control'
plot_dims = [0, 1, 2]  # For '2D', '3D', 'time' and 'error' plot types, specify which state dimensions to plot
nTraj = 3  # Number of trajectories to simulate and plot
disturbance_switch = True  # Add constant disturbance together with Gaussian forces to the payload and quadrotors
sigma = 0.3  # Standard deviation of Gaussian noise added; 0.3 is set for figure 8 in our paper
UDE_switch = True  # Enable UDE
attitude_tracking_switch = True  # Enable attitude tracking controller and dynamics of quadrotors
seed = 0  # Random seed for reproducibility

# Configuration variables
task = 'MUAV_point_mass'
pretrained = 'log_MUAV_point_mass_constrained'  # You'll need to set this to the path of your pretrained model
save_plot_path = os.path.join('results/plots', trajectory_type)  # Path to save the plot image, e.g., 'results/plots/3D_path.png'; to show the plot instead, set to None
save_csv_path = os.path.join('results/csvs', trajectory_type)  # Path to save the csv files; disable with None

# Create directory if not exist
if save_plot_path is not None:
    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)
if save_csv_path is not None:
    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure

np.random.seed(seed)

system = importlib.import_module('system_'+task)
f, B, B_w, _, num_dim_x, num_dim_control, num_dim_noise = get_system_wrapper(system)
controller = get_controller_wrapper(pretrained + '/controller_best.pth.tar')

if __name__ == '__main__':
    config = importlib.import_module('config_'+task)
    t = config.t
    time_bound = config.time_bound
    time_step = config.time_step
    XE_INIT_MIN = config.XE_INIT_MIN
    XE_INIT_MAX = config.XE_INIT_MAX

    x_0, xstar_0, ustar = config.system_reset(np.random.rand(), trajectory_type=trajectory_type)
    ustar = [u.reshape(-1,1) for u in ustar]
    xstar_0 = xstar_0.reshape(-1,1)
    xstar, _, _, _, _, _, _, _ = RK4_system(None, f, B, B_w, None, ustar, xstar_0, time_bound, time_step, with_tracking=False)

    fig = plt.figure(figsize=(8.0, 5.0))
    if plot_type=='3D':
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.gca()

    if plot_type == 'time':
        cmap = plt.get_cmap('plasma')
        colors = [cmap(i) for i in np.linspace(0, 1, len(plot_dims))]

    # States and controls of closed-loop simulation
    x_closed = []
    x_att_closed = []
    controls = []
    controls_C3M = []
    errors = []
    xinits = []
    # UDE results
    delta_j_bot_errs = []
    delta_T_errs = []
    for _ in range(nTraj):
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN) # Randomly sample XE_init
        # xe_0 = XE_INIT_MIN + np.round(np.random.rand(len(XE_INIT_MIN))) * (XE_INIT_MAX) * 2 # Only plot XE_init_min and XE_init_max
        xinit = xstar_0 + xe_0.reshape(-1,1)
        xinits.append(xinit)
        x, x_att, u_C3M, u, delta_j_bot_hat, delta_j_bot_err, delta_T_hat, delta_T_err = RK4_system(controller, f, B, B_w, xstar, ustar, xinit, 
                                                                                                    time_bound, time_step, sigma=sigma, with_tracking=True, 
                                                                                                    disturbance_switch=disturbance_switch, UDE_switch=UDE_switch, 
                                                                                                    attitude_tracking_switch=attitude_tracking_switch)
        x_closed.append(x)
        x_att_closed.append(x_att)
        controls.append(u)
        controls_C3M.append(u_C3M)
        delta_j_bot_errs.append(delta_j_bot_err)
        delta_T_errs.append(delta_T_err)

    for n_traj in range(nTraj):
        initial_dist = np.sqrt(((x_closed[n_traj][0] - xstar[0])**2).sum())
        errors.append([np.sqrt(((x-xs)**2).sum()) / initial_dist for x, xs in zip(x_closed[n_traj][:-1], xstar)])

        if plot_type=='2D':
            plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif plot_type=='3D':
            plt.plot([x[plot_dims[0],0] for x in x_closed[n_traj]], [x[plot_dims[1],0] for x in x_closed[n_traj]], [x[plot_dims[2],0] for x in x_closed[n_traj]], 'g', label='closed-loop traj' if n_traj==0 else None)
        elif plot_type=='time':
            for i, plot_dim in enumerate(plot_dims):
                plt.plot(t, [x[plot_dim,0] for x in x_closed[n_traj]][:-1], color=colors[i])
        elif plot_type=='error':
            plt.plot(t, [np.sqrt(((x-xs)**2).sum()) for x, xs in zip(x_closed[n_traj][:-1], xstar)], 'g')
        elif plot_type=='control':
            for i in range(len(controls[n_traj][0])):  # for each control dimension
                plt.plot(t, [u[i,0] for u in controls[n_traj]], label=f'u[{i}]')
                plt.plot(t, [u[i,0] for u in controls_C3M[n_traj]], '--', label=f'u_C3M[{i}]')
                
    if plot_type=='2D':
        plt.plot([x[plot_dims[0],0] for x in xstar], [x[plot_dims[1],0] for x in xstar], 'k', label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], 'ro', markersize=3.)
        plt.xlabel("x")
        plt.ylabel("y")
    elif plot_type=='3D':
        plt.plot([x[plot_dims[0],0] for x in xstar], [x[plot_dims[1],0] for x in xstar], [x[plot_dims[2],0] for x in xstar], 'k', label='Reference')
        plt.plot(xstar_0[plot_dims[0]], xstar_0[plot_dims[1]], xstar_0[plot_dims[2]], 'ro', markersize=3.)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    elif plot_type=='time':
        for plot_dim in plot_dims:
            plt.plot(t, [x[plot_dim,0] for x in xstar][:-1], 'k')
        plt.xlabel("t")
        plt.ylabel("x")
    elif plot_type=='error':
        plt.xlabel("t")
        plt.ylabel("error")
    elif plot_type=='control':
        for i in range(num_dim_control):
            plt.plot(t, [u[i,0] for u in ustar], 'k')
        plt.xlabel("t")
        plt.ylabel("u")
    
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(labels):
        plt.legend(frameon=True)  # Set legend position here
    if save_plot_path is not None:
        plt.savefig(save_plot_path+f'/{plot_type}.png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    #  Save the simulation data into csv files
    # Time array
    t_arr = np.expand_dims(t, axis=-1)
    # Reference and simulated states
    x_closed_arr = np.array(x_closed)[:, :-1]
    x_att_closed_arr = np.array(x_att_closed)[:, :-1]
    xstar_arr = np.array(xstar)[:-1]
    # Reference and simulated controls
    controls_C3M_arr = np.array(controls_C3M)
    controls_arr = np.array(controls)
    ustar_arr = np.array(ustar)
    # UDE results
    delta_j_bot_errs_arr = np.array(delta_j_bot_errs)
    delta_T_errs_arr = np.array(delta_T_errs)
    
    # Save data
    if save_csv_path is not None:
        for i in range(nTraj):       
            sim_i = np.concatenate((t_arr, np.squeeze(x_closed_arr[i]), np.squeeze(x_att_closed_arr[i]), np.squeeze(xstar_arr)), axis=1)  
            np.savetxt(save_csv_path + f'/sim_{i+1}.csv', sim_i, delimiter=',')
            print(f"Saved {save_csv_path}/sim_{i+1}.csv")
            con_i = np.concatenate((t_arr, np.squeeze(controls_C3M_arr[i]), np.squeeze(controls_arr[i]), np.squeeze(ustar_arr)), axis=1)
            np.savetxt(save_csv_path + f'/con_{i+1}.csv', con_i, delimiter=',')
            print(f"Saved {save_csv_path}/con_{i+1}.csv")
            ude_i = np.concatenate((t_arr, np.squeeze(delta_j_bot_errs_arr[i].T), delta_T_errs_arr[i].reshape(-1, 1)), axis = 1)
            np.savetxt(save_csv_path + f'/ude_{i+1}.csv', ude_i, delimiter=',')
            print(f"Saved {save_csv_path}/ude_{i+1}.csv")
    
    # sim_i columns: time, x_closed, x_att_closed, xstar
    # con_i columns: time, controls_C3M, controls, ustar
    # ude_i columns: time, delta_j_bot_errs, delta_T_errs

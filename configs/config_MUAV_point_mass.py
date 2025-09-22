import numpy as np
import math
from utils import temp_seed

### Training parameters ###

# Control bounds (force limits for each quadrotor) - increased for better control authority
f_bound = 3.0

# Cable projection constraint - sqrt(r_j_proj^2 + r_j_proj^2) must be less than 0.83
r_j_proj_x = 0.5
r_j_proj_y = 0.5

# Angles for gravity compensation
theta_z = math.radians(15)
theta_xy = math.radians(30)

# Mass of the payload
m_p = 1.3
m_q = [1.5, 1.5, 1.5]  # mass of the quadrotors

X_MIN = np.array([
    -10., -10., -10.,  # payload position bounds
    -r_j_proj_x, -r_j_proj_y, 
    -r_j_proj_x, -r_j_proj_y,
    -r_j_proj_x, -r_j_proj_y,  # cable projection bounds
    -2.5, -2.5, -2.5,  # payload velocity bounds
    -0.5, -0.5, -0.5, -0.5, -0.5, -0.5   # cable velocity bounds
]).reshape(-1,1)

X_MAX = np.array([
    10., 10., 10.,   # payload position bounds
    r_j_proj_x, r_j_proj_y, 
    r_j_proj_x, r_j_proj_y,
    r_j_proj_x, r_j_proj_y,   # cable projection bounds
    2.5, 2.5, 2.5,   # payload velocity bounds
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5   # cable velocity bounds
]).reshape(-1,1)


UREF = 1 / 3 * m_p * 9.81 * np.array([
                math.tan(theta_z), 0., 1.,  # Quadrotor 1: [delta_fx, delta_fy, delta_fz]
                - math.sin(theta_xy) * math.tan(theta_z), math.tan(theta_z) * math.cos(theta_xy), 1.,  # Quadrotor 2: [delta_fx, delta_fy, delta_fz]
                - math.sin(theta_xy) * math.tan(theta_z), -math.tan(theta_z) * math.cos(theta_xy), 1.   # Quadrotor 3: [delta_fx, delta_fy, delta_fz]
            ]).reshape(-1,1)
UREF_MIN = np.array([
    -f_bound, -f_bound, -f_bound,
    -f_bound, -f_bound, -f_bound,
    -f_bound, -f_bound, -f_bound
]).reshape(-1,1) + UREF

UREF_MAX = np.array([
    f_bound, f_bound, f_bound,
    f_bound, f_bound, f_bound,
    f_bound, f_bound, f_bound
]).reshape(-1,1) + UREF

# Error bounds for training
lim = 1.0

XE_MIN = np.array([
    -lim, -lim, -lim,
    -lim/3, -lim/3, -lim/3, -lim/3, -lim/3, -lim/3,
    -lim, -lim, -lim,
    -lim/6, -lim/6, -lim/6, -lim/6, -lim/6, -lim/6
]).reshape(-1,1)

XE_MAX = np.array([
    lim, lim, lim,
    lim/3, lim/3, lim/3, lim/3, lim/3, lim/3,
    lim, lim, lim,
    lim/6, lim/6, lim/6, lim/6, lim/6, lim/6
]).reshape(-1,1)


### Simulation parameters ###

# Circular path parameters
r_c = 2
v_c = 0.2
omega_c = 0.1

# Figure 8 path parameters
rx_8 = 3.0
ry_8 = 3.0
omega_8 = 0.1

# Hover initial condition
X_INIT_h = np.array([
    0., 0., 0.,  # payload position
    0.98 * math.sin(theta_z), 0., # Drone 1 cable position (r_1_x, r_1_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), 0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 2 cable position (r_2_x, r_2_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), -0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 3 cable position (r_3_x, r_3_y)
    0., 0., 0.,  # payload velocity
    0., 0., 0., 0., 0., 0.   # cable velocities
])

# Circular path initial condition
X_INIT_c = np.array([
    r_c, 0., 0.,  # payload position
    0.98 * math.sin(theta_z), 0., # Drone 1 cable position (r_1_x, r_1_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), 0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 2 cable position (r_2_x, r_2_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), -0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 3 cable position (r_3_x, r_3_y)
    0., r_c * omega_c, 0.,  # payload velocity
    0., 0., 0., 0., 0., 0.   # cable velocities
])

# Figure 8 path initial condition
X_INIT_8 = np.array([
    0., 0., 0.,  # payload position
    0.98 * math.sin(theta_z), 0.,  # Drone 1 cable position (r_1_x, r_1_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), 0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 2 cable position (r_2_x, r_2_y)
    -0.98 * math.sin(theta_xy) * math.sin(theta_z), -0.98 * math.sin(theta_z) * math.cos(theta_xy),  # Drone 3 cable position (r_3_x, r_3_y)
    rx_8 * omega_8, ry_8 * omega_8, 0.,  # payload velocity
    0., 0., 0., 0., 0., 0.   # cable velocities
])


# Initial error bounds for simulations
XE_INIT_MIN = np.array([-0.5, -0.5, -0.5,
                        -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                        -0.2, -0.2, -0.2,
                        -0.05, -0.05, -0.05, -0.05, -0.05, -0.05])

XE_INIT_MAX = np.array([0.5, 0.5, 0.5,
                        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.2, 0.2,
                        0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# Time parameters
time_bound = 63 # Circular path period = 63s, Figure-8 path period = 63s
time_step = 0.01
t = np.arange(0, time_bound, time_step)


def system_reset(seed, trajectory_type = 'hover'):
    SEED_MAX = 10000000
    if trajectory_type == 'hover':
        xref_0 = X_INIT_h
    elif trajectory_type == 'circular':
        xref_0 = X_INIT_c
    elif trajectory_type == 'figure-8':
        xref_0 = X_INIT_8
    else:
        raise NotImplementedError
    
    with temp_seed(int(seed * SEED_MAX)):
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        uref = []
        for _t in t:
            # Gravity of quadrotor is compensated in the system dynamics F_g term
            # uref is the additional acceleration needed for payload gravity and trajectory tracking

            # Payload gravity for hover
            a_h = 1 / 3 * m_p * 9.81 * np.array([
                math.tan(theta_z), 0., 1.,  # Quadrotor 1: [delta_fx, delta_fy, delta_fz]
                - math.sin(theta_xy) * math.tan(theta_z), math.tan(theta_z) * math.cos(theta_xy), 1.,  # Quadrotor 2: [delta_fx, delta_fy, delta_fz]
                - math.sin(theta_xy) * math.tan(theta_z), -math.tan(theta_z) * math.cos(theta_xy), 1.   # Quadrotor 3: [delta_fx, delta_fy, delta_fz]
                ])
            
            # Extra acceleration for circular path
            a_c = 1 / 3 * (m_p + sum(m_q)) * np.array([
                -r_c * omega_c**2 * math.cos(omega_c*_t), -r_c * omega_c**2 * math.sin(omega_c*_t), 0., 
                -r_c * omega_c**2 * math.cos(omega_c*_t), -r_c * omega_c**2 * math.sin(omega_c*_t), 0., 
                -r_c * omega_c**2 * math.cos(omega_c*_t), -r_c * omega_c**2 * math.sin(omega_c*_t), 0. ])
            
            # Extra acceleration for figure-8 path
            a_8 = 1 / 3 * (m_p + sum(m_q)) * np.array([
                -rx_8 * omega_8**2 * math.sin(omega_8*_t), -2 * ry_8 * omega_8**2 * math.sin(2*omega_8*_t), 0., 
                -rx_8 * omega_8**2 * math.sin(omega_8*_t), -2 * ry_8 * omega_8**2 * math.sin(2*omega_8*_t), 0., 
                -rx_8 * omega_8**2 * math.sin(omega_8*_t), -2 * ry_8 * omega_8**2 * math.sin(2*omega_8*_t), 0. ])
                
            if trajectory_type == 'hover':
                u = a_h
            elif trajectory_type == 'circular':
                u = a_h + a_c
            elif trajectory_type == 'figure-8':
                u = a_h + a_8
            else:
                raise NotImplementedError

            uref.append(u)

    return x_0, xref_0, uref

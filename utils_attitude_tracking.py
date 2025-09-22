import numpy as np
from collections import deque

def attitude_control_module(u, f_d_list_curr, attcurr, dt, i):
    # Attitude control module for 3 drones
    # Update list of desired force vectors
    f_d_1_list, f_d_2_list, f_d_3_list = f_d_list_curr
    f_d_1_list.append(u[0:3])  # Desired force vector for quadcopter 1
    f_d_2_list.append(u[3:6])  # Desired force vector for quadcopter 2
    f_d_3_list.append(u[6:9])  # Desired force vector for quadcopter 3
    # Get previous quaternions and angular velocities
    q_1_curr = attcurr[0:4,:]
    q_2_curr = attcurr[4:8,:]
    q_3_curr = attcurr[8:12,:]
    omega_1_curr = attcurr[12:15,:]
    omega_2_curr = attcurr[15:18,:]
    omega_3_curr = attcurr[18:21,:]
    # Attitude tracking controller for each quadcopter
    f_q1, q_1_next, omega_1_next = thrust_vector_dynamics(f_d_1_list, q_1_curr, omega_1_curr, dt, i)
    f_q2, q_2_next, omega_2_next = thrust_vector_dynamics(f_d_2_list, q_2_curr, omega_2_curr, dt, i)
    f_q3, q_3_next, omega_3_next = thrust_vector_dynamics(f_d_3_list, q_3_curr, omega_3_curr, dt, i)
    # Combine the forces and attitude states into a single output vector
    f_q = np.vstack((f_q1, f_q2, f_q3))
    attnext = np.vstack((q_1_next, q_2_next, q_3_next, omega_1_next, omega_2_next, omega_3_next))
    # Restore f_d_list to only keep last 2 elements
    f_d_1_list.pop()
    f_d_2_list.pop()
    f_d_3_list.pop()
    return f_q, attnext

def thrust_vector_dynamics(f_d_list, q_curr, omega_curr, dt, i):
    # f_d to f_q
            
    # Define constants
    m_q = 1.5  # mass of the quadrotor
    g_I = np.array([[0], [0], [-9.81]])  # gravity vector in inertial frame
    J_B = np.diag(np.array([0.1, 0.1, 0.3]))
    K_q = 100
    K_R = 0.5*np.eye(3)
    K_omega = 2.0*np.eye(3)

    # Define desired force vectors at time steps i-2, i-1, i
    # Calculate total desired thrust force by removing the gravity vector compensation
    f_d_p2 = f_d_list[0] - m_q * g_I
    f_d_p1 = f_d_list[1] - m_q * g_I
    f_d = f_d_list[2] - m_q * g_I
    T_d = np.linalg.norm(f_d) # Desired thrust magnitude

    R_IB_curr = quat_to_dcm(q_curr)
    R_IB_d, omega_d, omega_dot_d = attitude_thrust_extraction(f_d_p2, f_d_p1, f_d, dt, i)
    tau = attitude_tracker(R_IB_d, omega_d, omega_dot_d, R_IB_curr, omega_curr, J_B, K_R, K_omega)
    q_next, omega_next = attitude_dynamics(q_curr, omega_curr, tau, J_B, K_q, dt)
    R_IB_next = quat_to_dcm(q_next)
    f_q = R_IB_next @ np.array([[0], [0], [T_d]]) + m_q * g_I # Include gravity compensation
    return f_q, q_next, omega_next


def attitude_thrust_extraction(f_d_p2, f_d_p1, f_d, dt, i):
    # f_d = desired force vector at timestep k = T_d * n_d
    # f_d_p1 = desired force vector at timestep k-1
    # f_d_p2 = desired force vector at timestep k-2

    if i == 0:
        n_d = f_d / np.linalg.norm(f_d)
        psi = 0
        R_IB_d = rotation_matrix_extraction(n_d, psi)
        omega_d = np.array([[0], [0], [0]])
        omega_dot_d = np.array([[0], [0], [0]])
    elif i == 1:
        n_d = f_d / np.linalg.norm(f_d)
        n_d_p1 = f_d_p1/np.linalg.norm(f_d_p1)
        psi = 0
        R_IB_d = rotation_matrix_extraction(n_d, psi)
        R_IB_d_p1 = rotation_matrix_extraction(n_d_p1, psi)
        omega_d = get_omega(R_IB_d, R_IB_d_p1, dt)
        omega_dot_d = np.array([[0], [0], [0]])
    elif i >= 2:
        n_d = f_d / np.linalg.norm(f_d)
        n_d_p1 = f_d_p1/np.linalg.norm(f_d_p1)
        n_d_p2 = f_d_p2/np.linalg.norm(f_d_p2)
        psi = 0
        R_IB_d = rotation_matrix_extraction(n_d, psi)
        R_IB_d_p1 = rotation_matrix_extraction(n_d_p1, psi)
        R_IB_d_p2 = rotation_matrix_extraction(n_d_p2, psi)
        omega_d = get_omega(R_IB_d, R_IB_d_p1, dt)
        omega_d_p1 = get_omega(R_IB_d_p1, R_IB_d_p2, dt)
        omega_dot_d = get_omega_dot(omega_d, omega_d_p1, dt)
    return R_IB_d, omega_d, omega_dot_d


def attitude_tracker(R_IP_d, omega_d, omega_dot_d, R_IP, omega, J, K_R, K_omega):
    """
    Attitude tracking controller for a rigid body.

    Parameters:
    R_IP_d : Desired rotation matrix (3x3 numpy array)
    omega_d : Desired angular velocity (3x1 numpy array)
    omega_dot_d : Desired angular acceleration (3x1 numpy array)
    R_IP : Current rotation matrix (3x3 numpy array)
    omega : Current angular velocity (3x1 numpy array)
    J : Inertia matrix (3x3 numpy array)
    k_R : Gain for rotation error
    k_omega : Gain for angular velocity error

    Returns:
    tau : Control moment (3x1 numpy array)
    """
    # omega_d = np.zeros(omega_d.shape)
    # omega_dot_d = np.zeros(omega_dot_d.shape)
    R_tilde = np.transpose(R_IP_d) @ R_IP
    omega_tilde = omega - np.transpose(R_tilde) @ omega_d
    e_1 = np.array([1, 0, 0]).reshape(-1, 1)
    e_2 = np.array([0, 1, 0]).reshape(-1, 1)
    e_3 = np.array([0, 0, 1]).reshape(-1, 1)
    tau_d = -K_R @ (skew(e_1) @ R_tilde @ e_1 + skew(e_2) @ R_tilde @ e_2 + skew(e_3) @ R_tilde @ e_3) - K_omega @ omega_tilde # Desired torque
    tau_1 = -skew(omega_tilde) @ J @ omega_tilde
    tau_2 = skew(omega) @ J @ omega
    tau_3 = -J @ (skew(omega) @ np.transpose(R_tilde) @ omega_d - np.transpose(R_tilde) @ omega_dot_d)
    tau = tau_d + tau_1 + tau_2 + tau_3

    return tau

# Get quaternion and omega from dynamics
def attitude_dynamics(q_curr, omega_curr, tau, J, K_q, dt):
    # Update angular velocity
    omega_next = rk4_step(x = omega_curr, dt = dt, f = omega_b_dynamics, tau = tau, Inertia = J)
    # Update quarternion
    q_next = rk4_step(x = q_curr, dt = dt, f = Quaternion_Attitude_Dynamics, omega_b = omega_curr, K_q = K_q)
    q_next = quat_normalize(q_next)
    return q_next, omega_next

# Utilities for altitude_extraction
def get_omega(R_IB, R_IB_prev, dt):
    # Get omega from numerical integration of rotation matrices
    omega_cross = (np.matmul(np.transpose(R_IB_prev), R_IB) - np.eye(3))/dt
    omega_cross = (omega_cross - np.transpose(omega_cross))/2
    return vee_map(omega_cross)

def get_omega_dot(omega, omega_prev, dt):
    omega_dot = (omega - omega_prev)/dt
    return omega_dot

def rotation_matrix_extraction(n_d, psi):
    # R_IB = [n_x; n_y; n_z]
    n_z = n_d
    n_zx = n_z[0,0]
    n_zy = n_z[1,0]
    n_zz = n_z[2,0]
    n_x_tilde = np.array([[np.cos(psi)], [np.sin(psi)], [-(np.cos(psi)*n_zx + np.sin(psi)*n_zy)/n_zz]])
    n_x = n_x_tilde/np.linalg.norm(n_x_tilde)
    n_y = skew(n_z) @ n_x
    n_y = n_y/np.linalg.norm(n_y)
    R_IB = np.hstack([n_x, n_y, n_z])
    return R_IB


# Utilities for attitude dynamics
def quat_to_dcm(q):
    q = quat_normalize(q)
    R_IB = R_hat(q) @ np.transpose(L_hat(q))
    return R_IB


def omega_b_dynamics(omega_b, tau, Inertia):
    # Define Cross function for omega_b
    cross = skew(omega_b)
    # Define Dynamics
    Iwb = Inertia @ omega_b
    cross_term = cross @ Iwb
    return np.linalg.solve(Inertia, (tau - cross_term))


def Quaternion_Attitude_Dynamics(q, omega_b, K_q):
    q = quat_normalize(q)
    return 0.5*(np.transpose(L_hat(q)) @ omega_b) + ((K_q*(1-np.linalg.norm(q)))*q).reshape(-1, 1)
    

# Define Euler angle to quaternion calc
def euler_to_quat(roll, pitch, yaw):
        cr = np.cos(roll/2)
        sr = np.sin(roll/2)
        cp = np.cos(pitch/2)
        sp = np.sin(pitch/2)
        cy = np.cos(yaw/2)
        sy = np.sin(yaw/2)
        q0 = cr*cp*cy + sr*sp*sy
        q1 = sr*cp*cy - cr*sp*sy
        q2 = cr*sp*cy + sr*cp*sy
        q3 = cr*cp*sy - sr*sp*cy
        q = np.array([[q0], [q1], [q2], [q3]])
        return q

# Then Define L_hat and R_hat
def L_hat(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    L_hat = np.array([[-q1, q0, q3, -q2],
                    [-q2, -q3, q0, q1],
                    [-q3, q2, -q1, q0]])
    return L_hat


def R_hat(q):
    q0 = q[0,0]
    q1 = q[1,0]
    q2 = q[2,0]
    q3 = q[3,0]
    R_hat = np.array([[-q1, q0, -q3, q2],
                    [-q2, q3, q0, -q1],
                    [-q3, -q2, q1, q0]])
    return R_hat

# Normalize quaternion function
def quat_normalize(q):
    q.reshape(-1, 1)
    n = np.linalg.norm(q)
    return np.array([[1.0], [0.0], [0.0], [0.0]]) if n == 0 else q / n

# General Utils

# Generic RK4 Integrator step function
def rk4_step(x, dt, f, *args, **kwargs):
    k1 = f(x,               *args, **kwargs)
    k2 = f(x + 0.5*dt*k1,  *args, **kwargs)
    k3 = f(x + 0.5*dt*k2,  *args, **kwargs)
    k4 = f(x + dt*k3,      *args, **kwargs)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Skew symmetric matrix from vector
def skew(v):
    """
    Returns the skew-symmetric matrix of a 3D vector.
    """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

# Vee map from skew-symmetric matrix to vector
def vee_map(matrix):
    # Extracts the vector from a 3x3 skew-symmetric matrix.
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")
    if not np.allclose(matrix, -matrix.T):
        matrix = (matrix - matrix.T) / 2  # Make it skew-symmetric
    # Extract the vector components
    a = matrix[2, 1]
    b = matrix[0, 2]
    c = matrix[1, 0]
    return np.array([[a], [b], [c]])
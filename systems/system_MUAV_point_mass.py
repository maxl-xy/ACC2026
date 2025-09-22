import torch
import math
import numpy as np

num_dim_x = 18
num_dim_control = 9
num_dim_noise = 12

# Physical parameters
g = 9.81
m_p = 1.3  # mass of the payload
m_q = [1.5, 1.5, 1.5]  # mass of the quadrotors
l = [0.98, 0.98, 0.98]  # cable lengths


def get_Z(r, l):
    r_dot_product = torch.bmm(r.transpose(1, 2), r)
    Z = torch.sqrt(l**2 - r_dot_product)
    return Z


def get_B(r, l):
    bs = r.shape[0]
    Z = get_Z(r, l)
    I2 = torch.eye(2).repeat(bs, 1, 1).type(r.type())
    B = torch.cat([I2, -r.transpose(1, 2)/Z], dim=1)
    return B


def get_B_dot(r, v, l):
    bs = r.shape[0]
    O2 = torch.zeros(bs, 2, 2).type(r.type())
    Z = get_Z(r, l)
    r_T = r.transpose(1, 2)
    v_T = v.transpose(1, 2)
    B_dot = torch.cat([O2, -(Z ** 2 * v_T + torch.bmm(r_T, v) * r_T)/(Z ** 3)], dim=1)
    return B_dot


def f_func(x):
    """System dynamics"""
    bs = x.shape[0]
    
    # Extract states
    x_p_x, x_p_y, x_p_z, r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y = [x[:, i, 0] for i in range(9)]         # Payload position and cable projections
    v_p_x, v_p_y, v_p_z, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = [x[:, i, 0] for i in range(9, 18)]  # Payload and cable velocities

    # State vectors
    x_p = torch.stack([x_p_x, x_p_y, x_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    v_p = torch.stack([v_p_x, v_p_y, v_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    
    # Cable vectors
    r_1 = torch.stack([r_1_x, r_1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_2 = torch.stack([r_2_x, r_2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_3 = torch.stack([r_3_x, r_3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    v_q1 = torch.stack([v_q1_x, v_q1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q2 = torch.stack([v_q2_x, v_q2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q3 = torch.stack([v_q3_x, v_q3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    # Cable matrices
    B_1 = get_B(r_1, l[0])  # (bs, 3, 2)
    B_2 = get_B(r_2, l[1])  # (bs, 3, 2)
    B_3 = get_B(r_3, l[2])  # (bs, 3, 2)

    B_1_T = B_1.transpose(1, 2)  # (bs, 2, 3)
    B_2_T = B_2.transpose(1, 2)  # (bs, 2, 3)
    B_3_T = B_3.transpose(1, 2)  # (bs, 2, 3)

    B_1_dot = get_B_dot(r_1, v_q1, l[0])
    B_2_dot = get_B_dot(r_2, v_q2, l[1])
    B_3_dot = get_B_dot(r_3, v_q3, l[2])

    # Mass matrix M (9x9)
    M11 = (m_p + sum(m_q)) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M12 = m_q[0] * B_1
    M13 = m_q[1] * B_2
    M14 = m_q[2] * B_3
    M22 = m_q[0] * torch.bmm(B_1_T, B_1)
    M33 = m_q[1] * torch.bmm(B_2_T, B_2)
    M44 = m_q[2] * torch.bmm(B_3_T, B_3)

    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    
    M = torch.cat([
        torch.cat([M11, M12, M13, M14], dim=2),
        torch.cat([M12.transpose(1, 2), M22, zeros_22, zeros_22], dim=2),
        torch.cat([M13.transpose(1, 2), zeros_22, M33, zeros_22], dim=2),
        torch.cat([M14.transpose(1, 2), zeros_22, zeros_22, M44], dim=2)
    ], dim=1)

    # Coriolis matrix C (9x9)
    zeros_33 = torch.zeros(bs, 3, 3).type(x.type())
    zeros_23 = torch.zeros(bs, 2, 3).type(x.type())
    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    
    # C matrix components
    C12 = m_q[0] * B_1_dot
    C13 = m_q[1] * B_2_dot
    C14 = m_q[2] * B_3_dot
    C22 = m_q[0] * torch.bmm(B_1_T, B_1_dot) 
    C33 = m_q[1] * torch.bmm(B_2_T, B_2_dot)
    C44 = m_q[2] * torch.bmm(B_3_T, B_3_dot)

    C = torch.cat([
        torch.cat([zeros_33, C12, C13, C14], dim=2),
        torch.cat([zeros_23, C22, zeros_22, zeros_22], dim=2),
        torch.cat([zeros_23, zeros_22, C33, zeros_22], dim=2),
        torch.cat([zeros_23, zeros_22, zeros_22, C44], dim=2)
    ], dim=1)
    
    # Gravity forces
    g_I = torch.tensor([0, 0, -g]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())

    # Calculate z-axis angles
    #theta_1 = torch.asin(torch.norm(r_1)/l[0])
    #theta_2 = torch.asin(torch.norm(r_2)/l[1])
    #theta_3 = torch.asin(torch.norm(r_3)/l[2])

    # Balance the gravity forces with an angle theta
    #G_balance_1 = torch.tensor([torch.tan(theta_1), 0, 1]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())
    #G_balance_2 = torch.tensor([-0.5 * torch.tan(theta_2), torch.tan(theta_2) * torch.cos(torch.deg2rad(torch.tensor(30.0))), 1]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())
    #G_balance_3 = torch.tensor([-0.5 * torch.tan(theta_3), -torch.tan(theta_3) * torch.cos(torch.deg2rad(torch.tensor(30.0))), 1]).view(1, 3, 1).expand(bs, 3, 1).type(x.type())

    G1 = m_p * g_I
    G2 = torch.zeros(bs, 6, 1).type(x.type())

    F_g = torch.cat([G1, G2], dim=1)

    # Dynamics
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    
    # Kinematic equations (upper 14 states)
    vel = torch.cat([v_p, v_q1, v_q2, v_q3], dim=1)
    f[:, 0:9, 0] = vel.squeeze(-1)
    
    # Dynamic equations (lower 12 states)
    dynamics_term = torch.linalg.solve(M, F_g - torch.bmm(C, vel))
    f[:, 9:18, 0] = dynamics_term.squeeze(-1)
    return f


def DfDx_func(x):
    raise NotImplemented('NotImplemented')


def B_func(x):
    """Control input matrix (equivalent to gx from original class)"""
    bs = x.shape[0]
    
    # Extract states
    x_p_x, x_p_y, x_p_z, r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y = [x[:, i, 0] for i in range(9)]         # Payload position and cable projections
    v_p_x, v_p_y, v_p_z, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = [x[:, i, 0] for i in range(9, 18)]  # Payload and cable velocities

    # State vectors
    x_p = torch.stack([x_p_x, x_p_y, x_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    v_p = torch.stack([v_p_x, v_p_y, v_p_z], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    
    # Cable vectors
    r_1 = torch.stack([r_1_x, r_1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_2 = torch.stack([r_2_x, r_2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_3 = torch.stack([r_3_x, r_3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    v_q1 = torch.stack([v_q1_x, v_q1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q2 = torch.stack([v_q2_x, v_q2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    v_q3 = torch.stack([v_q3_x, v_q3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    # Cable matrices
    B_1 = get_B(r_1, l[0])  # (bs, 3, 2)
    B_2 = get_B(r_2, l[1])  # (bs, 3, 2)
    B_3 = get_B(r_3, l[2])  # (bs, 3, 2)

    B_1_T = B_1.transpose(1, 2)  # (bs, 2, 3)
    B_2_T = B_2.transpose(1, 2)  # (bs, 2, 3)
    B_3_T = B_3.transpose(1, 2)  # (bs, 2, 3)

    B_1_dot = get_B_dot(r_1, v_q1, l[0])
    B_2_dot = get_B_dot(r_2, v_q2, l[1])
    B_3_dot = get_B_dot(r_3, v_q3, l[2])
    
    # Mass matrix M (9x9)
    M11 = (m_p + sum(m_q)) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M12 = m_q[0] * B_1
    M13 = m_q[1] * B_2
    M14 = m_q[2] * B_3
    M22 = m_q[0] * torch.bmm(B_1_T, B_1)
    M33 = m_q[1] * torch.bmm(B_2_T, B_2)
    M44 = m_q[2] * torch.bmm(B_3_T, B_3)

    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    
    M = torch.cat([
        torch.cat([M11, M12, M13, M14], dim=2),
        torch.cat([M12.transpose(1, 2), M22, zeros_22, zeros_22], dim=2),
        torch.cat([M13.transpose(1, 2), zeros_22, M33, zeros_22], dim=2),
        torch.cat([M14.transpose(1, 2), zeros_22, zeros_22, M44], dim=2)
    ], dim=1)

    # Control input matrix H (9 x 9)
    H = torch.zeros(bs, 9, 9).type(x.type())

    H[:, 0:3, 0:3] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())
    H[:, 0:3, 3:6] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())
    H[:, 0:3, 6:9] = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())
    H[:, 3:5, 0:3] = B_1_T
    H[:, 5:7, 3:6] = B_2_T
    H[:, 7:9, 6:9] = B_3_T

    # B matrix: control input effect on full state
    control_effect = torch.linalg.solve(M, H)
    B = torch.cat([torch.zeros(bs, 9, num_dim_control).type(x.type()), control_effect], dim=1)
    return B


def DBDx_func(x):
    raise NotImplemented('NotImplemented')


def Bbot_func(B):
    # Compute Bbot
    Bbot = []
    for i in range(B.shape[0]):
        gi = B[i]  # num_dim_x x num_dim_control
        # SVD: Bi = U S Vh
        U, S, Vh = torch.linalg.svd(gi, full_matrices=True)
        # Null space: columns of U corresponding to zero singular values
        # For numerical stability, use a tolerance
        tol = 1e-9
        null_mask = S < tol
        if null_mask.sum() == 0:
            # If no exact zeros, take the last (n-m) columns of U
            Bbot_i = U[:, num_dim_control:]
        else:
            Bbot_i = U[:, null_mask]
        Bbot.append(Bbot_i)
    # Stack to shape bs x n x (n-m)
    Bbot = torch.stack(Bbot, dim=0).type(B.type())
    return Bbot


def B_w_func(x):
    """noise input to state"""
    bs = x.shape[0]
    # Extract states
    x_p_x, x_p_y, x_p_z, r_1_x, r_1_y, r_2_x, r_2_y, r_3_x, r_3_y = [x[:, i, 0] for i in range(9)]         # Payload position and cable projections
    v_p_x, v_p_y, v_p_z, v_q1_x, v_q1_y, v_q2_x, v_q2_y, v_q3_x, v_q3_y = [x[:, i, 0] for i in range(9, 18)]  # Payload and cable velocities
    
    # Cable vectors
    r_1 = torch.stack([r_1_x, r_1_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_2 = torch.stack([r_2_x, r_2_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)
    r_3 = torch.stack([r_3_x, r_3_y], dim=1).unsqueeze(-1)  # (bs, 2, 1)

    # Cable matrices
    B_1 = get_B(r_1, l[0])  # (bs, 3, 2)
    B_2 = get_B(r_2, l[1])  # (bs, 3, 2)
    B_3 = get_B(r_3, l[2])  # (bs, 3, 2)

    B_1_T = B_1.transpose(1, 2)  # (bs, 2, 3)
    B_2_T = B_2.transpose(1, 2)  # (bs, 2, 3)
    B_3_T = B_3.transpose(1, 2)  # (bs, 2, 3)

    # Mass matrix M (9x9)
    M11 = (m_p + sum(m_q)) * torch.eye(3).repeat(bs, 1, 1).type(x.type())
    M12 = m_q[0] * B_1
    M13 = m_q[1] * B_2
    M14 = m_q[2] * B_3
    M22 = m_q[0] * torch.bmm(B_1_T, B_1)
    M33 = m_q[1] * torch.bmm(B_2_T, B_2)
    M44 = m_q[2] * torch.bmm(B_3_T, B_3)

    zeros_22 = torch.zeros(bs, 2, 2).type(x.type())
    
    M = torch.cat([
        torch.cat([M11, M12, M13, M14], dim=2),
        torch.cat([M12.transpose(1, 2), M22, zeros_22, zeros_22], dim=2),
        torch.cat([M13.transpose(1, 2), zeros_22, M33, zeros_22], dim=2),
        torch.cat([M14.transpose(1, 2), zeros_22, zeros_22, M44], dim=2)
    ], dim=1)
    
    B_w = torch.zeros(bs, num_dim_x, num_dim_noise).type(x.type())
    
    I3 = torch.eye(3).unsqueeze(0).expand(bs, -1, -1).type(x.type())
    O23 = torch.zeros(bs, 2, 3).type(x.type())

    H_w = torch.cat([
        torch.cat([I3, I3, I3, I3], dim=2),
        torch.cat([O23, B_1_T, O23, O23], dim=2),
        torch.cat([O23, O23, B_2_T, O23], dim=2),
        torch.cat([O23, O23, O23, B_3_T], dim=2)
    ], dim=1)

    noise_effect = torch.linalg.solve(M, H_w)
    B_w = torch.cat([torch.zeros(bs, 9, num_dim_noise).type(x.type()), noise_effect], dim=1)

    return B_w

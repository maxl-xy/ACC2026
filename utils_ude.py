import numpy as np

# class SlungLoadPointMassUdeParam:
#     def __init__(self, m_p=1.3,
#                  m_q = [1.5, 1.5, 1.5],
#                  l=0.98):
#         self.m_p = m_p
#         self.m_q = m_q
#         self.l = l

# class SlungLoadPointMassUdeLog:
#     def __init__(self):
#         self.deltaJBotErr = list()
#         self.deltaTErr = np.array([])

# class SlungLoadPointMassUde:
#     def __init__(self, param: SlungLoadPointMassUdeParam):
#         self.param = param
#         self.n = len(param.m_q)
#         self.log = SlungLoadPointMassUdeLog()
#         self.quadSpeedIdxStart = list()
#         self.quadSpeedIdxEnd = list()
#         self.quadSpeedIdxStart = list()
#         self.quadSpeedIdxEnd = list()


#     def update_delta_j_bot(self):
#         return
#     def update_ude(self):
#         return self.mass * 9.81
    
#     def get_compensation_force(self, idx):
#         return 0.0

#     def get_log(self, angle):
#         return self.mass * 9.81 * np.cos(angle)

def GetCompensationForce(delta_j_bot_hat, delta_T_hat, time_idx, xcurr, l_j=0.98):
    res = np.zeros((9, 1))
    if time_idx <= 0:
        return res
    # for i in range(3):
    #     tmp = delta_j_bot_hat[i][time_idx - 1] + delta_T_hat[time_idx - 1] / 3.0
    #     start_idx, end_idx = GetControlForceIdx(i)
    #     res[start_idx:end_idx, 0] = tmp
    
    # get the cable vector
    A = np.zeros((3, 3))
    L_j_list = []
    for i in range(3):
        start_pos_idx, end_pos_idx = GetQuadPosIdx(i)
        r_j = np.reshape(xcurr[start_pos_idx:end_pos_idx], (2, 1))
        r_dot_product = float(r_j.T @ r_j)
        Z = np.sqrt(l_j**2 - r_dot_product)
        l_j_vec = np.vstack([r_j, Z])
        A[:, i] = l_j_vec.flatten() / l_j
        L_j_list.append(l_j_vec)
    delta_T_hat_drone = np.linalg.solve(A, delta_T_hat[time_idx - 1].reshape(-1, 1))

    for i in range(3):
        tmp = delta_j_bot_hat[i][time_idx - 1] + delta_T_hat_drone[i] * L_j_list[i].T
        start_idx, end_idx = GetControlForceIdx(i)
        res[start_idx:end_idx, 0] = tmp
    return res


def GetControlForceIdx(idx):
    return 3 * idx, 3 * idx + 3


def GetDroneDistErr(idx, dist, l_0_vec, delta_0_bot_hat):
    # get the disturbance estimation error
    start_drone_dist_idx, end_drone_dist_idx = GetDroneDisIdx(idx)
    delta_j = np.reshape(dist[start_drone_dist_idx:end_drone_dist_idx], (3, 1))
    delta_j_bot = GetDeltaBot(delta_j, l_0_vec)
    delta_j_p = delta_j - delta_j_bot
    return np.linalg.norm(delta_j_bot - delta_0_bot_hat), delta_j_bot, delta_j_p


def GetDroneDisIdx(idx):
    return 3 * idx + 3, 3 * idx + 6


def GetDeltaBot(delta_j, l_j, l=0.98):
    return (np.eye(3) - l_j @ l_j.T / l**2) @ delta_j

def UdeDroneDisturbanceUpdate(kappa, BB_j, d_f_L_j, v_q_j_dot, m_j, delta_j_hat):
    return kappa * BB_j @ (m_j * v_q_j_dot - d_f_L_j - delta_j_hat)

def GetQuadSpeedIdx(idx):
    # index starting from 0!
    return 2* idx + 12, 2* idx + 14

def GetQuadPosIdx(idx):
    return 2* idx + 3, 2* idx + 5

def GetQuadInertialVelAndAcc(v_j, v_j_dot, v_p_dot, B_j, B_j_dot):
    # get the idex of v_j:
    return v_p_dot + B_j_dot @ v_j + B_j @ v_j_dot

def GetUDEAuxMatrix(B_j):
    return B_j @ np.linalg.inv(B_j.T @ B_j) @ B_j.T

def GetUDEAux(r_j, v_j, l_j=0.98):
    r_dot_product = float(r_j.T @ r_j)
    Z = np.sqrt(l_j**2 - r_dot_product)
    l_j_vec = np.vstack([r_j, Z])
    I2 = np.eye(2)
    B_j = np.vstack([I2, -r_j.reshape(1, -1)/Z])
    
    O2 = np.zeros((2, 2))
    r_T_v = float(r_j.T @ v_j)
    B_j_dot_bottom = -(Z**2 * v_j.reshape(1, -1) + r_T_v * r_j.reshape(1, -1)) / (Z**3)
    B_j_dot = np.vstack([O2, B_j_dot_bottom])
    BB_j = GetUDEAuxMatrix(B_j)
    return B_j, B_j_dot, l_j_vec, BB_j

def calculate_quadrotor_kinematics(x_p, r_j, v_p, v_j, x_dot, v_j_dot, l_j=0.98):
    r_dot_product = np.dot(r_j, r_j)
    Z = np.sqrt(l_j**2 - r_dot_product)
    
    l_j_vec = np.array([r_j[0], r_j[1], Z])
    x_j = x_p + l_j_vec
    
    I2 = np.eye(2)
    B_j = np.vstack([I2, -r_j.reshape(1, -1)/Z])
    v_j_I = v_p + B_j @ v_j
    
    O2 = np.zeros((2, 2))
    r_T_v = np.dot(r_j, v_j)
    B_j_dot_bottom = -(Z**2 * v_j.reshape(1, -1) + r_T_v * r_j.reshape(1, -1)) / (Z**3)
    B_j_dot = np.vstack([O2, B_j_dot_bottom])
    
    v_p_dot = x_dot[9:12]
    a_j = v_p_dot + B_j_dot @ v_j + B_j @ v_j_dot
    
    return x_j, v_j_I, a_j
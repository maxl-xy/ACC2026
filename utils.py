import contextlib
import numpy as np
from collections import deque
from utils_attitude_tracking import *
from utils_ude import *

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def RK4_system(controller, f, B, B_w, xstar, ustar, xinit, t_max=10, dt=0.05, sigma=0., with_tracking=False, disturbance_switch=False, UDE_switch=False, attitude_tracking_switch=False):
    t = np.arange(0, t_max, dt)
    trace = []
    trace_att = []
    u = []
    u_C3M = []
    
    # Initilize states
    xcurr = xinit
    q_init = euler_to_quat(0, 0, 0) # Define initial attitude (roll, pitch, yaw)
    q_init = quat_normalize(q_init)
    omega_init = np.array([[0], [0], [0]]) # Define initial angular velocity
    attinit = np.vstack([q_init, q_init, q_init, omega_init, omega_init, omega_init])  # Initial attitude state vector for each quadcopter
    attcurr = attinit
    num_dim_noise = B_w(xcurr).shape[1]
    
    # ude results. if not active, return zero
    delta_j_hat = [np.zeros((t.size, 3)), np.zeros((t.size, 3)), np.zeros((t.size, 3))]
    delta_j_bot_hat = [np.zeros((t.size, 3)), np.zeros((t.size, 3)), np.zeros((t.size, 3))]
    delta_j_bot_err = [np.zeros((t.size)), np.zeros((t.size)), np.zeros((t.size))]
    
    delta_T_hat_int = np.zeros((t.size, 3)) # the integral term for delta_T estimation
    delta_T_hat = np.zeros((t.size, 3)) # the final delta_T estimation
    delta_T_err = np.zeros((t.size)) # the estimation error for delta_T
    
    # All intial values are set, ready to run with the RK4 integrator with the controller
    # Set Controller here, use this to generate the input to the system that then gets a force
    # generated for each quadcopter
    # Extract out attitude and implement the attitude tracker mechanism
    # Establish intial values of zero for the R and omega_b components
    # Attitude Dynamics update
    # Set initial variable for generic implementation

    trace.append(xcurr)
    trace_att.append(attcurr)

    # Initialize list of desired inputs (storing most current 3 inputs)
    f_d_init = np.array([[0], [0], [0]])
    f_d_1_list = deque([f_d_init, f_d_init, f_d_init])
    f_d_2_list = deque([f_d_init, f_d_init, f_d_init])
    f_d_3_list = deque([f_d_init, f_d_init, f_d_init])
    # Establish loop
    
    for i in range(len(t)):
        if with_tracking == False:
            ui = ustar[i]
            ui_C3M = np.zeros(ui.shape)
            # RK4 for ustar to generate xstar
            k1 = f(xcurr) + B(xcurr).dot(ui)
            k2 = f(xcurr + 0.5*dt*k1) + B(xcurr + 0.5*dt*k1).dot(ui)
            k3 = f(xcurr + 0.5*dt*k2) + B(xcurr + 0.5*dt*k2).dot(ui)
            k4 = f(xcurr + dt*k3) + B(xcurr + dt*k3).dot(ui)
            dx = (k1 + 2*k2 + 2*k3 + k4) / 6
            attnext = attcurr
        
        elif with_tracking == True:
            # Only consider disturbance, attitude tracking and UDE when tracking is on
            noise = np.zeros(num_dim_noise)
            if disturbance_switch == True:
                noise[0] = 0.3
                noise[1] = -0.2
                noise[2] = 0.5
                noise[3:12] = 0.3
                # Add small Gaussian noise
                noise += np.random.randn(num_dim_noise) * sigma
            noise = noise.reshape(-1, 1)
            
            # Update the desired force vector list if attitude tracking is on
            if attitude_tracking_switch == True:
                f_d_1_list.popleft()
                f_d_2_list.popleft()
                f_d_3_list.popleft()

            # RK4 step 1 
            xe = xcurr - xstar[i]
            ui = controller(xcurr, xe, ustar[i])
            ui_C3M = ui # CCM controller
            
            comp = np.zeros((9, 1))
            if UDE_switch == True:
                comp = GetCompensationForce(delta_j_bot_hat, delta_T_hat, i, xcurr)
                # comp = np.zeros((9, 1))
                # add compensation force to the control input
                ui = ui - comp
            ui_comp = ui # UDE compensated control input
            if attitude_tracking_switch == True:
                ui, attnext = attitude_control_module(ui, (f_d_1_list, f_d_2_list, f_d_3_list), attcurr, dt, i)  # Get the forces for each quadcopter
            k1 = f(xcurr) + B(xcurr).dot(ui) + B_w(xcurr).dot(noise)
            
            # RK4 step 2
            ui_k1 = controller(xcurr + 0.5*dt*k1, xe, ustar[i])
            if UDE_switch == True:
                ui_k1 = ui_k1 - comp
            if attitude_tracking_switch == True:
                ui_k1, _ = attitude_control_module(ui_k1, (f_d_1_list, f_d_2_list, f_d_3_list), attcurr, dt, i)  # Get the forces for each quadcopter
            k2 = f(xcurr + 0.5*dt*k1) + B(xcurr + 0.5*dt*k1).dot(ui_k1) + B_w(xcurr + 0.5*dt*k1).dot(noise)

            # RK4 step 3
            ui_k2 = controller(xcurr + 0.5*dt*k2, xe, ustar[i])
            if UDE_switch == True:
                ui_k2 = ui_k2 - comp   
            if attitude_tracking_switch == True:
                ui_k2, _ = attitude_control_module(ui_k2, (f_d_1_list, f_d_2_list, f_d_3_list), attcurr, dt, i)  # Get the forces for each quadcopter
            k3 = f(xcurr + 0.5*dt*k2) + B(xcurr + 0.5*dt*k2).dot(ui_k2) + B_w(xcurr + 0.5*dt*k2).dot(noise)

            # RK4 step 4
            ui_k3 = controller(xcurr + dt*k3, xe, ustar[i])
            if UDE_switch == True:
                ui_k3 = ui_k3 - comp             
            if attitude_tracking_switch == True:
                ui_k3, _ = attitude_control_module(ui_k3, (f_d_1_list, f_d_2_list, f_d_3_list), attcurr, dt, i)  # Get the forces for each quadcopter           
            k4 = f(xcurr + dt*k3) + B(xcurr + dt*k3).dot(ui_k3) + B_w(xcurr + dt*k3).dot(noise)
            dx = (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Update attitude states if attitude tracking is on
            if attitude_tracking_switch == True:
                # Update list of desired force vectors       
                f_d_1_list.append(ui_comp[0:3]) # Desired force vector for quadcopter 1
                f_d_2_list.append(ui_comp[3:6]) # Desired force vector for quadcopter 2
                f_d_3_list.append(ui_comp[6:9]) # Desired force vector for quadcopter 3
            else:
                # Keep attitude states the same if attitude tracking is off
                attnext = attcurr
                
            # update the ude for each quadcopter
            if UDE_switch and i > 1:
                x_dot = k1    
                mt = 1.3 + 3 * 1.5
                m_j = 1.5
                m_p = 1.3
                mBvJ = np.zeros((3, 1))
                totalF = np.zeros((3, 1))
                totalDeltaj = np.zeros((3, 1))
                totalDeltaj_true = np.zeros((3, 1))
                lambdaT = 1.0
                for idx in range(3):
                    # get the velocity and acceleration of the payload
                    v_p_dot = np.reshape(x_dot[9:12], (3, 1))
                    # get the quadrotor inertial velocity and acceleration
                    start_sp_idx, end_sp_idx = GetQuadSpeedIdx(idx)
                    start_pos_idx, end_pos_idx = GetQuadPosIdx(idx)
                    v_j = np.reshape(xcurr[start_sp_idx:end_sp_idx], (2, 1))
                    v_j_dot = np.reshape(x_dot[start_sp_idx:end_sp_idx], (2, 1))
                    r_j = np.reshape(xcurr[start_pos_idx:end_pos_idx], (2, 1))
                    B_j, B_j_dot, l_j_vec, BB_j = GetUDEAux(r_j, v_j, l_j=0.98)
                    v_q_dot_j = GetQuadInertialVelAndAcc(v_j, v_j_dot, v_p_dot, B_j, B_j_dot)
                    # get control force
                    start_con_idx, end_con_idx = GetControlForceIdx(idx)
                    delta_f_j = ui[start_con_idx: end_con_idx] # if ude is used, add ude to the control force

                    # update ude
                    dhat_j_pre = np.reshape(delta_j_hat[idx][i-1], (3, 1))
                    delta_j_dot = UdeDroneDisturbanceUpdate(1.0, BB_j, delta_f_j, v_q_dot_j, 1.5, dhat_j_pre)
                    # print("delta_0_dot:", delta_0_dot.T)
                    delta_j_hat[idx][i] = delta_j_hat[idx][i-1] + delta_j_dot.T*dt
                    tmp = GetDeltaBot(np.reshape(delta_j_hat[idx][i], (3, 1)), l_j_vec)
                    delta_j_bot_hat[idx][i] = np.squeeze(tmp)

                    # get the disturbance estimation error
                    delta_j_bot_err_j, _, delta_j_p = GetDroneDistErr(idx, noise, l_j_vec, tmp)
                    delta_j_bot_err[idx][i] = delta_j_bot_err_j
                    totalDeltaj_true += delta_j_p
                    # auxiliary variables for delta_T estimation
                    mBvJ += m_j * B_j @ v_j
                    totalF += np.reshape(delta_f_j, (3, 1))
                    totalDeltaj += np.reshape(delta_j_bot_hat[idx][i], (3, 1))
                    # totalDeltaj_true +=  GetDeltaBot(delta_j, l_0_vec)
                
                # calculate the delta_T
                v_p = np.reshape(xcurr[9:12], (3, 1))

                delta_T_int_dot = - lambdaT * (totalF + totalDeltaj + 
                                            np.reshape(delta_T_hat[i- 1], (3, 1)) +
                                            m_p * np.array([[0.0], [0.0], [-9.81]]))
                # print("delta_T_hat[i]:\n", delta_T_hat[i - 1])
                # print("totalF + totalDeltaj:\n", totalF + totalDeltaj)
                # print("delta_T_int_dot:\n", delta_T_int_dot)
                delta_T_hat_int_tmp = np.reshape(delta_T_hat_int[i-1], (3, 1)) + delta_T_int_dot*dt
                delta_T_hat_int[i] = np.squeeze(delta_T_hat_int_tmp)
                delta_T_hat[i] = np.squeeze(delta_T_hat_int_tmp + lambdaT * (mt * v_p + mBvJ))

                # get the estimation error
                totalDeltaT_true = totalDeltaj_true + np.reshape(noise[0:3], (3, 1))
                tmp2 = delta_T_hat[i] - totalDeltaT_true.T
                delta_T_err[i] = np.linalg.norm(tmp2)

        # RK4 state update
        xnext = xcurr + dx*dt
        
        # Store states and inputs
        trace.append(xnext)
        trace_att.append(attnext)
        u_C3M.append(ui_C3M)
        u.append(ui)
        
        # Iterate to next step
        xcurr = xnext
        attcurr = attnext
    
    return trace, trace_att, u_C3M, u, delta_j_bot_hat, delta_j_bot_err, delta_T_hat, delta_T_err

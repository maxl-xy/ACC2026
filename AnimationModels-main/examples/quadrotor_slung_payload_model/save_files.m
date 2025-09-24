%% Load the CSV file
traj = 1;
data_sim = table2array(readtable("sim_"+ traj +".csv"));
data_con = table2array(readtable("con_"+ traj +".csv"));
data_ude = table2array(readtable("ude_"+ traj +".csv"));

%% States
X_p = data_sim(:, 2:4); % payload position

l = 0.98; % cable_length

r1 = data_sim(:,5:6);
X_quad1_p = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad1_p(i,:) = [r1(i,:) sqrt(l^2-norm(r1(i,:)))];
end

r2 = data_sim(:,7:8);
X_quad2_p = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad2_p(i,:) = [r2(i,:) sqrt(l^2-norm(r2(i,:)))];
end

r3 = data_sim(:,9:10);
X_quad3_p = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad3_p(i,:) = [r3(i,:) sqrt(l^2-norm(r3(i,:)))];
end

q1 = data_sim(:,20:23);
RIB_1 = zeros(3,3, length(data_sim));
for i = 1:length(data_sim)
    RIB_1(:,:,i) = quat2rot(q1(i,:));
end

q2 = data_sim(:,24:27);
RIB_2 = zeros(3,3, length(data_sim));
for i = 1:length(data_sim)
    RIB_2(:,:,i) = quat2rot(q2(i,:));
end

q3 = data_sim(:,28:31);
RIB_3 = zeros(3,3, length(data_sim));
for i = 1:length(data_sim)
    RIB_3(:,:,i) = quat2rot(q3(i,:));
end

X_p_star = data_sim(:, 41:43); % payload position

r1_star = data_sim(:,44:45);
X_quad1_p_star = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad1_p_star(i,:) = [r1_star(i,:) sqrt(l^2-norm(r1_star(i,:)))];
end

r2_star = data_sim(:,46:47);
X_quad2_p_star = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad2_p_star(i,:) = [r2_star(i,:) sqrt(l^2-norm(r2_star(i,:)))];
end

r3_star = data_sim(:,48:49);
X_quad3_p_star = zeros(length(data_sim),3); % payload to quadrotor 1 vector
for i = 1:length(data_sim)
    X_quad3_p_star(i,:) = [r3_star(i,:) sqrt(l^2-norm(r3_star(i,:)))];
end

%% Controls
u_C3M_1 = data_con(:,2:4);
u_C3M_2 = data_con(:,5:7);
u_C3M_3 = data_con(:,8:10);
u_1 = data_con(:,11:13);
u_2 = data_con(:,14:16);
u_3 = data_con(:,17:19);
u_star_1 = data_con(:,20:22);
u_star_2 = data_con(:,23:25);
u_star_3 = data_con(:,26:28);

%% UDE performance
delta_bot_err_1 = data_ude(:,2);
delta_bot_err_2 = data_ude(:,3);
delta_bot_err_3 = data_ude(:,4);
delta_T_err = data_ude(:,5);

%% Save data in matlab data file
% States
multiquadPayloadData.X1 = X_p + X_quad1_p;
multiquadPayloadData.X2 = X_p + X_quad2_p;
multiquadPayloadData.X3 = X_p + X_quad3_p;
multiquadPayloadData.PX = X_p;
multiquadPayloadData.RIB1 = RIB_1;
multiquadPayloadData.RIB2 = RIB_2;
multiquadPayloadData.RIB3 = RIB_3;
multiquadPayloadData.X1_star = X_p_star + X_quad1_p_star;
multiquadPayloadData.X2_star = X_p_star + X_quad2_p_star;
multiquadPayloadData.X3_star = X_p_star + X_quad3_p_star;
multiquadPayloadData.PX_star = X_p_star;
multiquadPayloadData.time = data_sim(:,1);
% Controls
multiquadPayloadData.UC3M1 = u_C3M_1;
multiquadPayloadData.UC3M2 = u_C3M_2;
multiquadPayloadData.UC3M3 = u_C3M_3;
multiquadPayloadData.U1 = u_1;
multiquadPayloadData.U2 = u_2;
multiquadPayloadData.U3 = u_3;
multiquadPayloadData.U1_star = u_star_1;
multiquadPayloadData.U2_star = u_star_2;
multiquadPayloadData.U3_star = u_star_3;
% UDE performance
multiquadPayloadData.DBE1 = delta_bot_err_1;
multiquadPayloadData.DBE2 = delta_bot_err_2;
multiquadPayloadData.DBE3 = delta_bot_err_3;
multiquadPayloadData.DTE = delta_T_err;

% Save the data as a .mat file
final_xUDE = multiquadPayloadData; % Change name of struct variable
save("final_xUDE.mat", 'final_xUDE'); % Save struct variable into .mat file 


%% Quaternion to rotation matrix function
function R_IB = quat2rot(q)
    % Ensure q is normalized
    q = q / norm(q);
    
    q0 = q(1); % scalar part
    q1 = q(2);
    q2 = q(3);
    q3 = q(4);
    
    L_hat = [-q1, q0, q3, -q2; 
             -q2, -q3, q0, q1;
             -q3, q2, -q1, q0];
    R_hat = [-q1, q0, -q3, q2;
             -q2, q3, q0, -q1;
             -q3, -q2, q1, q0];

    R_IB = R_hat * L_hat';
end

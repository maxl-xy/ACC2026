% ------------------------------------------------------------------------------
% MIT License
% 
% Copyright (c) 2023 Dr. Longhao Qian
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
% ------------------------------------------------------------------------------
% generate the animation of a quadrotor with a slung payload
close all
clear;
%% load the simulation data
% _cir for circular path; _ate for figure-8 path 
load("final.mat");
multiquadPayloadData = final;
load("final_xUDE.mat")
multiquadPayloadData_xUDE = final_xUDE;
addpath('../../src/models')
addpath('../../src/utils')
addpath('../../src/camera')
addpath('../../src/hud')
%% load flight path results
anim_fig = figure('Units', 'normalized', 'OuterPosition', [0.2 0.2 0.6 0.8]);
anim_fig.Renderer = 'Painters';
ax = axes('Position',[0.1 0.1 0.8 0.8],  'XLim',[-4, 4],'YLim',[-4, 4],'ZLim',[-3 3], 'DataAspectRatio', [1 1 1]);
% if in NED mode, reverse the plot
% set(ax, 'YDir', 'reverse')
% set(ax, 'ZDir', 'reverse')
xlabel(ax, 'x axis (m)','FontSize',16,'Interpreter','latex');
ylabel(ax, 'y axis (m)','FontSize',16,'Interpreter','latex');
zlabel(ax, 'z axis (m)','FontSize',16,'Interpreter','latex');
grid(ax, 'on')
hold on
%t1 = title(ax, 't = 0s');
%% draw trajectory
% Simulated
trajectory_quad_1 = line(multiquadPayloadData.X1(:, 1), multiquadPayloadData.X1(:, 2), multiquadPayloadData.X1(:, 3), 'LineWidth', 0.55, 'color', 'blue');
trajectory_quad_2 = line(multiquadPayloadData.X2(:, 1), multiquadPayloadData.X2(:, 2), multiquadPayloadData.X2(:, 3), 'LineWidth', 0.55, 'color', 'blue');
trajectory_quad_3 = line(multiquadPayloadData.X3(:, 1), multiquadPayloadData.X3(:, 2), multiquadPayloadData.X3(:, 3), 'LineWidth', 0.55, 'color', 'blue');
trajectory_payload = line(multiquadPayloadData.PX(:, 1), multiquadPayloadData.PX(:, 2), multiquadPayloadData.PX(:, 3), 'LineWidth', 1.1, 'color', 'green');
% Reference
% ref_traj_quad_1 = line(multiquadPayloadData.X1_star(:, 1), multiquadPayloadData.X1_star(:, 2), multiquadPayloadData.X1_star(:, 3), 'LineWidth', 0.75, 'color', 'black', 'LineStyle','-.');
% ref_traj_quad_2 = line(multiquadPayloadData.X2_star(:, 1), multiquadPayloadData.X2_star(:, 2), multiquadPayloadData.X2_star(:, 3), 'LineWidth', 0.75, 'color', 'black', 'LineStyle','-.');
% ref_traj_quad_3 = line(multiquadPayloadData.X3_star(:, 1), multiquadPayloadData.X3_star(:, 2), multiquadPayloadData.X3_star(:, 3), 'LineWidth', 0.75, 'color', 'black', 'LineStyle','-.');
ref_traj_payload = line(multiquadPayloadData.PX_star(:, 1), multiquadPayloadData.PX_star(:, 2), multiquadPayloadData.PX_star(:, 3), 'LineWidth', 1.75, 'color', 'black');
% Trajectory without UDE
trajectory_payload_xUDE = line(multiquadPayloadData_xUDE.PX(1:40:end, 1), multiquadPayloadData_xUDE.PX(1:40:end, 2), multiquadPayloadData_xUDE.PX(1:40:end, 3), 'LineWidth', 1.1, 'color', 'red', 'LineStyle', ':');
%% draw the quadrotor and the gates
quadObj_1 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
quadObj_2 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
quadObj_3 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
%% draw the payload
payloadObj = CreateSphere(0.08, 10, 'cyan', ax);
%% draw the cable
trajectory_cable_1 = line([multiquadPayloadData.X1(1, 1)  multiquadPayloadData.PX(1, 1)],...
                           [multiquadPayloadData.X1(1, 2) multiquadPayloadData.PX(1, 2)],...
                           [multiquadPayloadData.X1(1, 3) multiquadPayloadData.PX(1, 3)], 'LineWidth', 1.5, 'color', 'black');
trajectory_cable_2 = line([multiquadPayloadData.X2(1, 1)  multiquadPayloadData.PX(1, 1)],...
                           [multiquadPayloadData.X2(1, 2) multiquadPayloadData.PX(1, 2)],...
                           [multiquadPayloadData.X2(1, 3) multiquadPayloadData.PX(1, 3)], 'LineWidth', 1.5, 'color', 'black');
trajectory_cable_3 = line([multiquadPayloadData.X3(1, 1)  multiquadPayloadData.PX(1, 1)],...
                           [multiquadPayloadData.X3(1, 2) multiquadPayloadData.PX(1, 2)],...
                           [multiquadPayloadData.X3(1, 3) multiquadPayloadData.PX(1, 3)], 'LineWidth', 1.5, 'color', 'black');

%% record the gif
saveToGif = false;
filename_gif = "quadrotor_payload.gif";
frameRate = 20;
%% down sample the simulation data
idxArray = GetDownSampledIdx(1/frameRate, multiquadPayloadData.time, 1, length(multiquadPayloadData.time)-1);
n = length(idxArray);
itr = floor(n/5);
for i = 1:itr:itr*5 % 1:length(idxArray) 
    %% get the sampled idx
    k = idxArray(i);
    %% update the model
    % Quadrotor 1
    Reb_1 = reshape(multiquadPayloadData.RIB1(:, :, k), 3, 3);
    R_1 = [Reb_1 zeros(3, 1);
        zeros(1, 3), 1];
    xp_1 = multiquadPayloadData.X1(k, 1);
    yp_1 = multiquadPayloadData.X1(k, 2);
    zp_1 = multiquadPayloadData.X1(k, 3);
    x_1 = [xp_1, yp_1, zp_1];
    T_1 = makehgtform('translate', x_1');
    set(quadObj_1.frame, 'Matrix', T_1 * R_1);
    % Quadrotor 2
    Reb_2 = reshape(multiquadPayloadData.RIB2(:, :, k), 3, 3);
    R_2 = [Reb_2 zeros(3, 1);
        zeros(1, 3), 1];
    xp_2 = multiquadPayloadData.X2(k, 1);
    yp_2 = multiquadPayloadData.X2(k, 2);
    zp_2 = multiquadPayloadData.X2(k, 3);
    x_2 = [xp_2, yp_2, zp_2];
    T_2 = makehgtform('translate', x_2');
    set(quadObj_2.frame, 'Matrix', T_2 * R_2);
    % Quadrotor 3
    Reb_3 = reshape(multiquadPayloadData.RIB3(:, :, k), 3, 3);
    R_3 = [Reb_3 zeros(3, 1);
        zeros(1, 3), 1];
    xp_3 = multiquadPayloadData.X3(k, 1);
    yp_3 = multiquadPayloadData.X3(k, 2);
    zp_3 = multiquadPayloadData.X3(k, 3);
    x_3 = [xp_3, yp_3, zp_3];
    T_3 = makehgtform('translate', x_3');
    set(quadObj_3.frame, 'Matrix', T_3 * R_3);
    % Payload
    set(payloadObj.frame, 'Matrix', makehgtform('translate', multiquadPayloadData.PX(k, :)'));
    %% update the cable position
    set(trajectory_cable_1, 'XData', [multiquadPayloadData.X1(k, 1)  multiquadPayloadData.PX(k, 1)],...
        'YData', [multiquadPayloadData.X1(k, 2)  multiquadPayloadData.PX(k, 2)],...
        'ZData', [multiquadPayloadData.X1(k, 3)  multiquadPayloadData.PX(k, 3)]);
    set(trajectory_cable_2, 'XData', [multiquadPayloadData.X2(k, 1)  multiquadPayloadData.PX(k, 1)],...
        'YData', [multiquadPayloadData.X2(k, 2)  multiquadPayloadData.PX(k, 2)],...
        'ZData', [multiquadPayloadData.X2(k, 3)  multiquadPayloadData.PX(k, 3)]);
    set(trajectory_cable_3, 'XData', [multiquadPayloadData.X3(k, 1)  multiquadPayloadData.PX(k, 1)],...
        'YData', [multiquadPayloadData.X3(k, 2)  multiquadPayloadData.PX(k, 2)],...
        'ZData', [multiquadPayloadData.X3(k, 3)  multiquadPayloadData.PX(k, 3)]);
    %% update the camera
    % UpdateCameraModelSideView(ax, [0.5, 0.2, 0], [3, 2.5, 1], 3) % Circular
    UpdateCameraModelSideView(ax, [0.5, 0.2, 0.1], [4, 2.5, 1], 3) % Figure 8
    % UpdateCameraModelSideView(ax, x_1 + [0, 0, 0.8], [1.5, 1.5, 1.8], 3)
    %% update the title
    %set(t1, 'String', ['t=', num2str(multiquadPayloadData.time(k), '%.2f') ,'s']);
    %% update the plot
    drawnow % visually update the window on every iteration
    %% plot ghost objects
    ghost = true;
    if mod(i-1,itr) == 0 && ghost == true  % only every 10th frame
        % create ghost objects
        % Quadrotors
        ghostQuad_1 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
        set(ghostQuad_1.frame, 'Matrix', T_1 * R_1);
        ghostQuad_2 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
        set(ghostQuad_2.frame, 'Matrix', T_2 * R_2);
        ghostQuad_3 = CreateQuadRotor(0.25*0.55, 0.1*0.55, ax, 'ENU');
        set(ghostQuad_3.frame, 'Matrix', T_3 * R_3);
        % Payload
        ghostPayload = CreateSphere(0.08, 10, 'cyan', ax);
        set(ghostPayload.frame, 'Matrix', makehgtform('translate', multiquadPayloadData.PX(k, :)'));
        % Cables
        trajectory_cable_1 = line([multiquadPayloadData.X1(k, 1)  multiquadPayloadData.PX(k, 1)],...
                           [multiquadPayloadData.X1(k, 2) multiquadPayloadData.PX(k, 2)],...
                           [multiquadPayloadData.X1(k, 3) multiquadPayloadData.PX(k, 3)], 'LineWidth', 1.5, 'color', 'black');
        trajectory_cable_2 = line([multiquadPayloadData.X2(k, 1)  multiquadPayloadData.PX(k, 1)],...
                           [multiquadPayloadData.X2(k, 2) multiquadPayloadData.PX(k, 2)],...
                           [multiquadPayloadData.X2(k, 3) multiquadPayloadData.PX(k, 3)], 'LineWidth', 1.5, 'color', 'black');
        trajectory_cable_3 = line([multiquadPayloadData.X3(k, 1)  multiquadPayloadData.PX(k, 1)],...
                           [multiquadPayloadData.X3(k, 2) multiquadPayloadData.PX(k, 2)],...
                           [multiquadPayloadData.X3(k, 3) multiquadPayloadData.PX(k, 3)], 'LineWidth', 1.5, 'color', 'black');
    end

    %% Generate the GIF
    if saveToGif
        frame = getframe(anim_fig);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        if i == 1
            imwrite(imind,cm,filename_gif,'gif', 'Loopcount',inf, 'DelayTime', 1/frameRate);
        else
            imwrite(imind,cm,filename_gif,'gif','WriteMode','append', 'DelayTime', 1/frameRate);
        end  
    end
   
end

legend([trajectory_quad_1 trajectory_payload trajectory_payload_xUDE ref_traj_payload], ...
    'Quadrotor trajectory, UDE on','Payload trajectory, UDE on','Payload trajectory, UDE off','Reference payload trajectory', 'Interpreter','latex', 'Fontsize', 16, 'Location', 'northwest')

%% UDE plots
UDE = figure('Units', 'normalized', 'OuterPosition', [0.2 0.2 0.5 0.6]);
UDE.Renderer = 'Painters';
subplot(2,1,1)
xlim([0,63])
grid on
errors = vecnorm(multiquadPayloadData.PX - multiquadPayloadData.PX_star,2,2);
errors_xUDE = vecnorm(multiquadPayloadData_xUDE.PX - multiquadPayloadData_xUDE.PX_star,2,2);
line(multiquadPayloadData.time(3:end), errors(3:end), 'LineWidth', 1, 'Color', 'green')
line(multiquadPayloadData.time(3:end), errors_xUDE(3:end), 'LineWidth', 1, 'Color', 'red')
xlabel('Time (s)','Fontsize',14,'Interpreter','latex')
ylabel('$||$\boldmath{$\underline{x}$}$_p-$\boldmath{$\underline{x}$}$_p^*||$','Fontsize',14,'Interpreter','latex')
legend('UDE on','UDE off','Fontsize',14,'interpreter', 'latex')
subplot(2,1,2)
xlim([0,63])
grid on
line(multiquadPayloadData.time(3:end), multiquadPayloadData.DTE(3:end), 'LineWidth', 1, 'Color', 'blue')
line(multiquadPayloadData.time(3:end), multiquadPayloadData.DBE1(3:end), 'LineWidth', 1, 'Color', '#FF8800')
line(multiquadPayloadData.time(3:end), multiquadPayloadData.DBE2(3:end), 'LineWidth', 1, 'Color', '#FF8800')
line(multiquadPayloadData.time(3:end), multiquadPayloadData.DBE3(3:end), 'LineWidth', 1, 'Color', '#FF8800')
%title('UDE Performance in Figure-8 Trajectory Tracking', 'Interpreter','latex')
xlabel('Time (s)','Fontsize',14,'Interpreter','latex')
ylabel('Noise estimation error','Fontsize',14,'Interpreter','latex')
legend('$\tilde{\delta}_{T}$','$\tilde{\delta}_{\perp,1}$','$\tilde{\delta}_{\perp,2}$','$\tilde{\delta}_{\perp,3}$','Fontsize',14,'interpreter', 'latex')
% function pos, vel, u_input = generate_data()
    % Simulate MSD System - Healthy and Faulty Data
    clear; clc;
    
    M = 1; % 5 mass blocks
    fs = 1e4; % sampling freq = 10K
    f = 4800;    % freq of continues time signal
    T = 100;
    time = 0:1/fs:T;   % Simulation time
    x0 = zeros(2*M, 1); % Initial conditions: all positions/velocities zero
    x0(1) = 0;                % Small initial displacement on first mass
    w_cap = 2*pi*(f/fs);        % digital frequency
    u_input = @(n) 5*sin(w_cap*n); % External force on last mass (can be changed)
    % u_input = @(n) (n >= 0 && n < 0.0005) * 1000;
    
    nl_index.springs = []; %Nonlinear spring connecting m1 and m2
    nl_index.springs_wall = []; %Nonlinear spring connecting m3 and m4 to the wall
    nl_index.dampers = []; %Nonlinear damper between m1 and m2, and m2 and m3
    nl_index.dampers_wall = [];
    k_nl = @(x) 0;                     % @(x) 0.5*x+3*x^3; 
    d_nl = @(x,xdot) 0;               % @(x,xdot) xdot; 
    k_wall_nl = @(x) 0;                    % @(x) 0.5*x+3*x^3;
    d_wall_nl = @(x,xdot) 0;               % @(x,xdot) xdot; 
    
    % Parameter set: [mass, k0, d0, k0_wall, d0_wall]
    param_healthy = [0.3, 2, 0.5, 0.2, 0];
    param_faulty = [1, 0.1, 0.01, 0.1, 2]; % Simulate a fault: weaker spring and damping
    
    % Generate system dynamics
    [f_fun, x_sym, u_sym] = generalized_msd(M, nl_index, k_nl, d_nl, k_wall_nl, d_wall_nl);
    
    %% Simulate Healthy System
    f_healthy = @(t, x) f_fun(x, u_input(t), param_healthy);
    [t_healthy, x_healthy] = ode45(f_healthy, time, x0);
    
    %% Simulate Faulty System
    f_faulty = @(t, x) f_fun(x, u_input(t), param_faulty);
    [t_faulty, x_faulty] = ode45(f_faulty, time, x0);

    
    %% Plot Results 

    % (ADD UP THE POS SIGNALS for EACH MASS => P1 + P2 + P3 ..., Thats my 1 signal from the MSD COMPONENT)

    figure;
    for i = 1:M
        subplot(M,2,2*i-1)
        plot(t_healthy, x_healthy(:,i), 'b', 'LineWidth', 1.2);
        hold on;
        plot(t_faulty, x_faulty(:,i), 'r--', 'LineWidth', 1.2);
        title(['Position of Mass ', num2str(i)]);
        ylabel('Position (m)');
        legend('Healthy', 'Faulty');
    
        subplot(M,2,2*i)
        plot(t_healthy, x_healthy(:,M+i), 'b', 'LineWidth', 1.2);
        hold on;
        plot(t_faulty, x_faulty(:,M+i), 'r--', 'LineWidth', 1.2);
        title(['Velocity of Mass ', num2str(i)]);
        ylabel('Velocity (m/s)');
        legend('Healthy', 'Faulty');
    end
    xlabel('Time (s)');




function [pos, vel, acc, time, conn_pairs] = run_dynamics(machine_type, scenario, ds_type, ds_subtype)

    % Get root directory where run_dynamics.m is located
    root_dir_path = mfilename('fullpath');
    root_dir = fileparts(root_dir_path);

    % Target folder where generate_machine() lives
    target_dir = fullfile(root_dir, "machines", machine_type, scenario, ds_type, ds_subtype);

    % Add necessary paths
    addpath( root_dir);        % for generate machine to find generalized_msd 
    addpath(target_dir);   % to find generate_machine itself

    [f, conn, M] = generate_machine();

    % INPUT FORCE
    u_cells = cell(M, 1);
    for i = 1:M
        u_cells{i} = @(t) 0;
    end
    
    % Here, add desired forces for each mass
    
    % impulse force
    % u_cells{M} = @(t) (t >= 0 && t < 0.0005) * 1000; 
    
    % periodic force
    A = [1];    % amplitude
    f0 = 100;     % fundamental freq (hz)
    freq = [f0]; % freq (hz)
    
    u_cells{M} = @(t) A*sin(2*pi*freq*t) + 1e-7*randn(size(t));
    
    u = @(t) cellfun(@(f) f(t), u_cells);  % returns M×1 vector at time t
    
    % Params
    param = [conn.mass;
             conn.spring_k;
             conn.damper_d;
             conn.k_wall_lin;
             conn.d_wall_lin];
    
    odefun = @(t, x) f(x, u(t), param);
    
    % =======================================================
    
    % Time
    fs = 500;
    t_end = 1000;
    tspan = 0:1/fs:t_end;
    
    % Initial positions and velocities
    q0 = zeros(M,1);     % All at rest position
    qd0 = zeros(M,1);    % All stationary
    x0 = [q0; qd0];      % Initial state vector
    
    [time, X] = ode45(odefun, tspan, x0);

    %% Extract outputs
    
    % get connection pairs
    conn_pairs = conn.pairs;
    
    % Get pos, vel and acc from X

    pos = X(:, 1:M);         % size: [length(tspan) x M]
    vel = X(:, M+1:end);     % size: [length(tspan) x M]
    
    % compute acceleration 
    acc = zeros(size(pos));  % size: [length(tspan) x M]
    for i = 1:length(tspan)
        x_i = X(i, :)';                     % current state vector (pos, vel)
        t_i = tspan(i);                     
        dxdt = f(x_i, u(t_i), param);       % evaluate derivative
        acc(i, :) = dxdt(M+1:end)';         % second half of dxdt is acceleration
    end


    % === position plot ===
    figure;
    plot(tspan, pos, 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend(arrayfun(@(i) sprintf('Mass %d', i), 1:M, 'UniformOutput', false));
    title('Mass Positions Over Time');
    grid on;
    
    % === velocity plot ===
    figure;
    plot(tspan, vel, 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Velocity (m/s)');
    legend(arrayfun(@(i) sprintf('Mass %d', i), 1:M, 'UniformOutput', false));
    title('Mass Velocities Over Time');
    grid on;
    
    % === acceleration plot ===
    figure;
    plot(tspan, acc, 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Acceleration (m/s^2)');
    legend(arrayfun(@(i) sprintf('Mass %d', i), 1:M, 'UniformOutput', false));
    title('Mass Accelerations Over Time');
    grid on;
    
    % === plot inputs ===
    u_signal = zeros(length(time), M);
    for i = 1:length(time)
        u_signal(i, :) = u(time(i))';  % transpose to make it 1 × M
    end
    
    % find which masses have non-zero input
    non_zero_forces = find(cellfun(@(f) ~isequal(func2str(f), func2str(@(t) 0)), u_cells));
    
    figure;
    plot(time, u_signal(:, non_zero_forces), 'LineWidth', 1.5);
    xlabel("Time (s)");
    ylabel("Input Force (N)");
    legend(arrayfun(@(i) sprintf('Mass %d', i), non_zero_forces, 'UniformOutput', false));
    title('Input Forces Applied');
    grid on;
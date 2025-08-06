function [pos, acc, vel, ds_type] = run_dynamics()
    clear;
    [f, conn, M] = generate_healthy_machine();
    
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
    f0 = 1;     % fundamental freq (hz)
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
    fs = 1000;
    t_end = 100;
    tspan = 0:1/fs:t_end;
    
    % Initial positions and velocities
    q0 = zeros(M,1);     % All at rest position
    qd0 = zeros(M,1);    % All stationary
    x0 = [q0; qd0];      % Initial state vector
    
    [time, X] = ode45(odefun, tspan, x0);
    
    
    % Plot positions
    figure;
    plot(time, X(:,1:M), 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend(arrayfun(@(i) sprintf('Mass %d', i), 1:M, 'UniformOutput', false));
    title('Mass Displacements Over Time');
    grid on;
    
    % plot inputs
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
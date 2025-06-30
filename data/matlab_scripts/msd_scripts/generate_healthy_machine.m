function [h_dynamics, conn, M] = generate_healthy_machine()
    M = 8;

    %% Define connections / topology
    conn.pairs = [
        8 7;
        7 6;
        6 5;
        5 4; 5 1;
        4 3;
        3 2;
        2 1
        ];  % order: source of input u -> target

    %% Parameters

    % If you want to change the default values of parameters, set
    % config_param = true. 

    % If you want default values, set
    % config_param = false
    
    config_param = true;

    % Default paramter values
    conn.mass = ones(M,1);

    if size(conn.pairs, 1) > 0
        conn.spring_k = 0.5 * ones(size(conn.pairs, 1), 1);
        conn.damper_d = 0.5 * ones(size(conn.pairs, 1), 1);
    else
        conn.spring_k = 0;
        conn.damper_d = 0;
    end

    conn.k_wall_lin = 0 * ones(M, 1);
    conn.d_wall_lin = 0 * ones(M, 1);

    conn.k_wall_lin(1) = 0.5;
    conn.d_wall_lin(1) = 0.5;
    
    % Non linear
    nl_index.springs = [2]; % 2nd spring nonlinear
    nl_index.dampers = [];
    nl_index.springs_wall = [];
    nl_index.dampers_wall = [];
    
    k_nl = @(x) 0.5*x+3*x^3; 
    d_nl = @(x,xdot) xdot; 
    k_wall_nl = @(x) 0.5*x+3*x^3;
    d_wall_nl = @(x,xdot) xdot; 


    % Change default parameter values

    if config_param == true
        % to configure the mass values, linear spring/damper constants of the healthy machine, go to
        % config_healthy_machine_param function
        conn = config_healthy_machine_param(conn);
    end
    
    [h_dynamics, ~, ~, ~] = generalized_msd_2(M, conn, nl_index, k_nl, d_nl, k_wall_nl, d_wall_nl);

end
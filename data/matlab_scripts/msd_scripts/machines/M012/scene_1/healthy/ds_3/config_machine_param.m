function conn = config_machine_param(conn)

    % Configure the [individual mass values], 
    % [individiual spring/damper constants] of the healthy machine   
    % -------------------------------------------------------

    %% configure mass (kg)

    conn.mass(1) = 0.00011;
    conn.mass(2) = 0.0001;
    conn.mass(3) = 0.00014;
    conn.mass(4) = 0.0001;
    conn.mass(5) = 0.0001;
    conn.mass(6) = 0.00013;
    conn.mass(7) = 0.0001;
    conn.mass(8) = 0.0001;
    conn.mass(9) = 0.00017;
    conn.mass(10) = 0.0001;
    conn.mass(11) = 0.00018;
    conn.mass(12) = 0.0001;
    
    %% Spring/dampers between masses

    % configure linear spring
    conn.spring_k(1) = 0.501;
    conn.spring_k(2) = 0.5;
    conn.spring_k(3) = 0.498;
    conn.spring_k(4) = 0.502;
    conn.spring_k(5) = 0.5;
    conn.spring_k(6) = 0.5;
    conn.spring_k(7) = 0.508;
    conn.spring_k(8) = 0.5;
    conn.spring_k(9) = 0.5;
    
    
    % configure linear damper
    conn.damper_d(1) = 0.5;
    conn.damper_d(2) = 0.5;
    conn.damper_d(3) = 0.49;
    conn.damper_d(4) = 0.5;
    conn.damper_d(5) = 0.49;
    conn.damper_d(6) = 0.5;
    conn.damper_d(7) = 0.505;
    conn.damper_d(8) = 0.5;
    conn.damper_d(9) = 0.509;

    %% Springs/dampers between wall & masses
    
    % configure linear wall spring
    conn.k_wall_lin(1) = 0.01;
    conn.k_wall_lin(2) = 0;
    conn.k_wall_lin(3) = 0;
    conn.k_wall_lin(4) = 0;
    conn.k_wall_lin(5) = 0;
    conn.k_wall_lin(6) = 0.01;
    conn.k_wall_lin(7) = 0;
    conn.k_wall_lin(8) = 0;
    conn.k_wall_lin(9) = 0.01;
    conn.k_wall_lin(10) = 0;
    conn.k_wall_lin(11) = 0.01;
    conn.k_wall_lin(12) = 0;

    % configure linear wall damper
    conn.d_wall_lin(1) = 0;
    conn.d_wall_lin(2) = 0;
    conn.d_wall_lin(3) = 0;
    conn.d_wall_lin(4) = 0;
    conn.d_wall_lin(5) = 0;
    conn.d_wall_lin(6) = 0;
    conn.d_wall_lin(7) = 0;
    conn.d_wall_lin(8) = 0;
    conn.d_wall_lin(9) = 0;
    conn.d_wall_lin(10) = 0;
    conn.d_wall_lin(11) = 0;
    conn.d_wall_lin(12) = 0;



end
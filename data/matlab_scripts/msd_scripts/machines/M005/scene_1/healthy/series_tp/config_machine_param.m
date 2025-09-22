function conn = config_machine_param(conn)

    % Configure the [individual mass values], 
    % [individiual spring/damper constants] of the healthy machine   
    % -------------------------------------------------------

    %% configure mass (kg)

    conn.mass(1) = 0.0001;
    conn.mass(2) = 0.0001;
    conn.mass(3) = 0.0001;
    conn.mass(4) = 0.0001;
    conn.mass(5) = 0.0001;
    
    %% Spring/dampers between masses

    % configure linear spring
    conn.spring_k(1) = 1;
    conn.spring_k(2) = 1;
    conn.spring_k(3) = 1;
    
    
    % configure linear damper
    conn.damper_d(1) = 0;
    conn.damper_d(2) = 0;
    conn.damper_d(3) = 0;

    %% Springs/dampers between wall & masses
    
    % configure linear wall spring
    conn.k_wall_lin(1) = 0.01;
    conn.k_wall_lin(2) = 0.01;
    conn.k_wall_lin(3) = 0;
    conn.k_wall_lin(4) = 0;
    conn.k_wall_lin(5) = 0;

    % configure linear wall damper
    conn.d_wall_lin(1) = 0;
    conn.d_wall_lin(2) = 0;
    conn.d_wall_lin(3) = 0;
    conn.d_wall_lin(4) = 0;
    conn.d_wall_lin(5) = 0;



end
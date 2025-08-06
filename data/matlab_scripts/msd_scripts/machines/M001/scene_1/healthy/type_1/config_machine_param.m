function conn = config_machine_param(conn)

    % Configure the [individual mass values], 
    % [individiual spring/damper constants] of the healthy machine   
    % -------------------------------------------------------

    %% configure mass (kg)

    conn.mass(1) = 1;
    
    %% Spring/dampers between masses

    % configure linear spring
    conn.spring_k(1) = 0.5;
    
    % configure linear damper
    conn.damper_d(1) = 0;
    conn.damper_d(2) = 0;
    conn.damper_d(3) = 0;

    %% Springs/dampers between wall & masses
    
    % configure linear wall spring
    conn.k_wall_lin(1) = 0.5;
    conn.k_wall_lin(2) = 0;
    conn.k_wall_lin(3) = 0;
    conn.k_wall_lin(4) = 0;

    % configure linear wall damper
    conn.d_wall_lin(1) = 0;
    conn.d_wall_lin(2) = 0;
    conn.d_wall_lin(3) = 0;
    conn.d_wall_lin(4) = 0;



end
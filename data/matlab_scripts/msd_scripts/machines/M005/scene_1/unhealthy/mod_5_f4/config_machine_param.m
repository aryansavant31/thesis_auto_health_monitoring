function conn = config_machine_param(conn)

    % Configure the [individual mass values], 
    % [individiual spring/damper constants] of the healthy machine   
    % -------------------------------------------------------

    %% configure mass (kg)

    m1_nominal = 1;
    m2_nominal = 0.8;
    m3_nominal = 0.5;
    m4_nominal = 1;
    m5_nominal = 0.5;
    
    m_rand_range = 0.1; % 10% varaition

    conn.mass(1) = m1_nominal .* (1 + m_rand_range .* (2*rand(size(m1_nominal)) - 1));
    conn.mass(2) = m2_nominal .* (1 + m_rand_range .* (2*rand(size(m2_nominal)) - 1));
    conn.mass(3) = m3_nominal .* (1 + m_rand_range .* (2*rand(size(m3_nominal)) - 1));
    conn.mass(4) = m4_nominal .* (1 + m_rand_range .* (2*rand(size(m4_nominal)) - 1));
    conn.mass(5) = m5_nominal .* (1 + m_rand_range .* (2*rand(size(m5_nominal)) - 1));
    
    %% Spring/dampers between masses

    % configure linear spring
    k1_nominal = 15000;
    k2_nominal = 10000;
    k3_nominal = 15000;

    conn.spring_k(1) = k1_nominal .* (1 + 0.1 .* (2*rand(size(k1_nominal)) - 1));
    conn.spring_k(2) = k2_nominal .* (1 + 0.13 .* (2*rand(size(k2_nominal)) - 1));
    conn.spring_k(3) = k3_nominal .* (1 + 0.77 .* (2*rand(size(k3_nominal)) - 1));
    
    
    % configure linear damper
    c1_nominal = 4;
    c2_nominal = 3;
    c3_nominal = 4;

    conn.damper_d(1) = c1_nominal .* (1 + 0.3 .* (2*rand(size(c1_nominal)) - 1));
    conn.damper_d(2) = c2_nominal .* (1 + 0.3 .* (2*rand(size(c2_nominal)) - 1));
    conn.damper_d(3) = c3_nominal .* (1 + 0.85 .* (2*rand(size(c3_nominal)) - 1));

    %% Springs/dampers between wall & masses
    
    % configure linear wall spring
    k1_wall_nominal = 20000;
    k4_wall_nominal = 20000;

    conn.k_wall_lin(1) = k1_wall_nominal .* (1 + 0.12 .* (2*rand(size(k1_wall_nominal)) - 1));
    conn.k_wall_lin(2) = 0;
    conn.k_wall_lin(3) = 0;
    conn.k_wall_lin(4) = k4_wall_nominal .* (1 + 0.11 .* (2*rand(size(k4_wall_nominal)) - 1));
    conn.k_wall_lin(5) = 0;

    % configure linear wall damper
    c1_wall_nominal = 5;
    c4_wall_nominal = 5;

    conn.d_wall_lin(1) = c1_wall_nominal .* (1 + 0.3 .* (2*rand(size(c1_wall_nominal)) - 1));
    conn.d_wall_lin(2) = 0;
    conn.d_wall_lin(3) = 0;
    conn.d_wall_lin(4) = c4_wall_nominal .* (1 + 0.3 .* (2*rand(size(c4_wall_nominal)) - 1));
    conn.d_wall_lin(5) = 0;



end
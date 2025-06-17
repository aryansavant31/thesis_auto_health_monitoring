function [f, dfdxu, x, u] = generalized_msd_2(M, conn, nl_index, k, d, k_wall, d_wall)

    % ---------- ARGUMENT VALIDATION ----------
    arguments
        M (1,1) {mustBeInteger, mustBePositive}
        conn struct  % Connection structure
        nl_index struct
        k function_handle
        d function_handle
        k_wall function_handle
        d_wall function_handle
    end

    P = size(conn.pairs, 1);
    
    % ---------- [NEW BLOCK A] VALIDATE CONNECTION STRUCTURE ----------
    % conn should contain:
    % - mass:       [M x 1] vector of masses
    % - spring_k:   [P x 1] linear spring constants for P connections
    % - damper_d:   [P x 1] linear damper constants for P connections
    % - pairs:      [P x 2] each row (i,j) is a spring/damper between mass i and mass j
    % - k_wall_lin: [M x 1] wall spring constants
    % - d_wall_lin: [M x 1] wall damper constants
    
    assert(length(conn.mass) == M, "conn.mass must have M elements.");

    if M > 1
        assert(size(conn.pairs,2) == 2, "conn.pairs must be a Px2 array of mass connections.");
        assert(all(size(conn.spring_k) == size(conn.damper_d)) && ...
           size(conn.spring_k,1) == size(conn.pairs,1), ...
           "conn.spring_k/damper_d must match the number of pairs.");
    end
    
    
    % ---------- [NEW BLOCK B] SYMBOLIC DEFINITIONS ----------
    syms u [M 1] real
    q = sym('q',[M,1],'real');
    qd = sym('qd',[M,1],'real');
    m = sym('m', [M,1], 'real');

    kl_sym = sym('k', [P 1]);  % springs → k1, k2, ..., kN
    dl_sym = sym('d', [P 1]);  % dampers → d1, d2, ..., dN
    kwl_sym = sym('kw', [M 1]);    % wall springs → kw1, kw2, ...
    dwl_sym = sym('dw', [M 1]);   % wall dampers → dw1, dw2, ...

    param = [m; kl_sym; dl_sym; kwl_sym; dwl_sym];


    % ---------- [NEW BLOCK C] NONLINEAR SPRINGS/DAMPERS ----------

    knonlin = cell(P,1); knonlin(:) = {@(x) 0};
    dnonlin = cell(P,1); dnonlin(:) = {@(x,xdot) 0};
    wall_knonlin = cell(M,1); wall_knonlin(:) = {@(x) 0};
    wall_dnonlin = cell(M,1); wall_dnonlin(:) = {@(x,xdot) 0};
    
    knonlin(nl_index.springs) = {k};
    dnonlin(nl_index.dampers) = {d};
    wall_knonlin(nl_index.springs_wall) = {k_wall};
    wall_dnonlin(nl_index.dampers_wall) = {d_wall};
    
    % ---------- [MODIFIED BLOCK D] EQUATIONS OF MOTION ----------
    qdd = sym(zeros(M,1));
    for i = 1:M
        Ftot = 0;

        if P > 0
        % Forces b/w masses via sprins/damper
            for j = 1:P
                a = conn.pairs(j,1); % a = 2
                b = conn.pairs(j,2); % b = 1
                kij = conn.spring_k(j);
                dij = conn.damper_d(j);
                
                % Connection from i to another mass
                if a == i || b == i
                    % sign_ = (i == a) - (i == b); % 1 if i=a, -1 if i=b   %sign = 1
                    other = b * (i == a) + a * (i == b); % other = b = 1
        
                    dq = q(i) - q(other);   % dq = q2 - q1
                    dqd = qd(i) - qd(other); % dqd = qd2 - qd1
                    Fspring = Fs(dq, kij, knonlin{j});  %Fspring = k1 * (q2 - q1)
                    Fdamp = Fd(dq, dqd, dij, dnonlin{j}); % Fdamp = c1 * (qd2 - qd1)
                    Ftot = Ftot - Fspring - Fdamp;  % Ftot2 = - k1 * (q2 - q1) - c1 * (qd2 - qd1) - Fswall - Fdwall       
                end
            end
        end
    
        % Wall connection
        Fwall_s = Fs(q(i), conn.k_wall_lin(i), wall_knonlin{i});
        
        Fwall_d = Fd(q(i), qd(i), conn.d_wall_lin(i), wall_dnonlin{i});
        
        Ftot = Ftot - Fwall_s - Fwall_d + u(i);
    
        % Equation of motion
        qdd(i) = simplify(Ftot / conn.mass(i), 'Steps', 10);
    end

    
    % ---------- OUTPUT FORMULATION ----------
    x = [q; qd];
    f_ = [qd; qdd];
    f = matlabFunction(f_, 'Vars', {x, u, param});
    dfdxu_ = jacobian(f_, [x; u]);
    dfdxu = matlabFunction(dfdxu_, 'Vars', {x, u, param});
    end
    
    %%%%%%%%%%%%%%%%%%% LOCAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
    function y = Fs(x, klin, knonlin)
        y = klin * x + knonlin(x);
    end
    
    function y = Fd(x, xdot, dlin, dnonlin)
        y = dlin * xdot + dnonlin(x, xdot);
end

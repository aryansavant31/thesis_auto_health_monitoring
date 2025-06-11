function [f, dfdxu, x,u] = generalized_msd(M,nl_index,k,d, k_wall, d_wall)
%GENERALIZED_MSD Generates the equations of motion of a custom
%Mass-Spring-Damper system. Each mass is interconnected with the adjacent
%mass blocks, and to an infinitely rigid wall. Each connection contains a
%linear and a nonlinear spring and damper. The linear parameters and
%nonlinear functions must be defined by the user. The external force input
%"u" is applied to the last mass block M.
%
%   Syntax:
%       [f,x,u] = generalized_msd(M,nl_index,k,d, k_wall, d_wall) generates
%       the dynamic equations of the MSD system.
%
%   Inputs:
%       M (double): Amount of mass blocks.
%       nl_index (struct): Structure that must contain the following fields:
%           'springs' (array): Array containing the indices that define the positions where the
%              nonlinear springs between mass blocks are placed, \in {1, ..., M-1}
%           'dampers' (array): Array containing the indices that define the positions where the
%              nonlinear dampers between mass blocks are placed, \in {1, ..., M-1}
%           'springs_wall' (array): Array containing the indices that define the positions where the
%              nonlinear springs connecting a mass block with the wall are placed, \in {1, ..., M}
%           'dampers_wall' (array): Array containing the indices that define the positions where the
%              nonlinear dampers connecting a mass block with the wall are placed, \in {1, ..., M}
%       k (function_handle): Function of the form "k = @(x) f(x)"
%              that describes the nonlinear spring connecting mass blocks.
%       d (function_handle): Function of the form "d = @(x, xdot) f(x, xdot)"
%              that describes the nonlinear damper connecting mass blocks.
%       k_wall (function_handle): Function of the form "k_wall = @(x) f(x)"
%              that describes the nonlinear spring connecting a mass blocks with the wall.
%       d_wall (function_handle): Function of the form "d_wall = @(x, xdot) f(x, xdot)"
%              that describes the nonlinear damper connecting a mass blocks with the wall.
%
%   Outputs:
%       f: function_handle with handles @(x,u,params) that contains the
%           dynamic equations of the system and x is the state, u is the
%           input, and params is the array "params = [m k0 d0 k_wall d_wall]",
%           where:
%             m:  the mass (kg) of each mass block
%             k0: the linear spring constant (N/m) connecting mass blocks
%             d0: the linear damping constant (N/(m·s)) connecting mass blocks
%             k_wall: the linear spring constant (N/m) connecting mass and wall
%             d_wall: the linear damper constant (N/(m·s)) connecting mass and wall
%       dfdxu: function_handle with handles @(x,u,params) that provides the
%           Jacobian of the system and x is the state, u is the input, and 
%           params is the array "params = [m k0 d0 k_wall d_wall]".
%       x: symbolic vector containing the state variables.
%       u: symbolic vector containing the input variables.
%
%   EXAMPLE
%       M = 5; % 5 mass blocks
%       nl_index.springs = [1]; %Nonlinear spring connecting m1 and m2
%       nl_index.springs_wall = [3,4]; %Nonlinear spring connecting m3 and
%       m4 to the wall
%       nl_index.dampers = [1,2]; %Nonlinear damper between m1 and m2, and
%       m2 and m3
%       nl_index.dampers_wall = [];
%       k = @(x) 0.5*x+3*x^3; 
%       d = @(x,xdot) xdot; 
%       k_wall = @(x) 0.5*x+3*x^3;
%       d_wall = @(x,xdot) xdot; 
%       param = [1 0.5 1 0.5 1];
%       [fMSD,state,input] = generalized_msd(M,nl_index,k,d,k_wall,d_wall);
%       NONLINEAR DYN MODEL:
%       f = @(x,u) fMSD(x,u,param);
%       h = @(x,u) x(M,:); %Output chosen as the position of the last mass
% 
% Written by: Javi Olucha

    arguments
        M (1,1) {mustBeNumeric,mustBeFinite,mustBePositive}
        nl_index
        k function_handle
        d function_handle
        k_wall function_handle
        d_wall function_handle
    end

    nl_k_idx = nl_index.springs;
    nl_d_idx = nl_index.dampers;

    nl_k_wall_idx = nl_index.springs_wall;
    nl_d_wall_idx = nl_index.dampers_wall;

    if ~isempty(nl_k_idx) && (max(nl_k_idx) >= M || min(nl_k_idx) <= 0)
        error("Invalid choice of nonlinear springs indices (between blocks). Values must be between 1 and M-1.");
    end
    if ~isempty(nl_k_wall_idx) && (max(nl_k_wall_idx) > M || min(nl_k_wall_idx) <= 0)
        error("Invalid choice of nonlinear spring indices (connected to wall). Values must be between 1 and M.")
    end

    if ~isempty(nl_d_idx) && (max(nl_d_idx) >= M || min(nl_d_idx) <= 0)
        error("Invalid choice of nonlinear damper indices (between blocks). Values must be between 1 and M-1.")
    end
    if ~isempty(nl_d_wall_idx) && (max(nl_d_wall_idx) > M || min(nl_d_wall_idx) <= 0)
        error("Invalid choice of nonlinear damper indices (connected to wall). Values must be between 1 and M.")
    end
    %% Function init
    syms m k0 d0 k0_wall d0_wall real
    syms u real
    q = sym('q',[M,1],'real');
    qd = sym('qd',[M,1],'real');
    param = [m k0 d0 k0_wall d0_wall];

    %Nonlinear spring and dampers between masses
    klin = k0;
    knonlin = cell(M-1,1);
    knonlin(:) = {@(x) 0};
    knonlin(nl_k_idx) = {k};

    dlin = d0;
    dnonlin = cell(M-1,1);
    dnonlin(:) = {@(x, xdot) 0};
    dnonlin(nl_d_idx) = {d};

    %Nonlinear spring and dampers connecting each mass block to an infinitely rigid wall
    wall_klin = k0_wall;
    wall_knonlin = cell(M,1);
    wall_knonlin(:) = {@(x) 0};
    wall_knonlin(nl_k_wall_idx) = {k_wall};

    wall_dlin = d0_wall;
    wall_dnonlin = cell(M,1);
    wall_dnonlin(:) = {@(x, xdot) 0};
    wall_dnonlin(nl_d_wall_idx) = {d_wall};

    %%
    %Equations of motion
    qdd = sym(NaN(M,1));
    for i=1:M
        if i==1 && M==1
            % Special Case: Only a single block
            Ftot = u;
        elseif i==1
            % Total force on mass
            Ftot = Fs(q(2)-q(1),klin,knonlin{1}) + Fd(q(2)-q(1),qd(2)-qd(1), dlin,dnonlin{1});
        elseif i==M
            % Total force on mass
            Ftot = u - Fs(q(M)-q(M-1),klin,knonlin{M-1}) - Fd(q(M)-q(M-1), qd(M)-qd(M-1),dlin,dnonlin{M-1});
        else
            % Total force on mass
            Ftot = - Fs(q(i)-q(i-1),klin,knonlin{i-1}) - Fd(q(i)-q(i-1), qd(i)-qd(i-1),dlin,dnonlin{i-1}) ...
                + Fs(q(i+1)-q(i),klin,knonlin{i}) + Fd(q(i+1)-q(i), qd(i+1)-qd(i),dlin,dnonlin{i});
        end

        % Connect each block by an additional spring/damper to wall
        Ftot = Ftot - Fs(q(i),wall_klin,wall_knonlin{i}) - Fd(q(i),qd(i),wall_dlin,wall_dnonlin{i});

        % Acceleration
        qdd(i) = Ftot/m;
    end
    %%
    qdd = simplify(qdd,10);
    x = [q; qd];
    f_ = [qd; qdd];

    dfdxu_ = jacobian(f_, [x; u]);
    % dfdu_ = jacobian(f_, u);


    f = matlabFunction(f_,'Vars',{x,u,param});
    % dfdx = matlabFunction(dfdx_,'Vars',{x,u,param});
    % dfdu = matlabFunction(dfdu_,'Vars',{x,u,param});
    dfdxu = matlabFunction(dfdxu_,'Vars',{x,u,param});

end

%%%%%%%%%%%%%%%%%%% LOCAL FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%
%% Local function to compute spring force
function y = Fs(x,klin,knonlin)
    y = klin*x + knonlin(x);
end
%% Local function to compute damping force
function y = Fd(x,xdot,dlin,dnonlin)
    y = dlin*xdot + dnonlin(x,xdot);
end
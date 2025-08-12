t_end = 10;
ts = 0.1;
time = 0:ts:t_end;

A_sig = @(t) sin(2*pi*1*t);
B_sig = @(t) sin(2*pi*2*t);
C_sig = @(t) sin(2*pi*3*t);
D_sig = @(t) sin(2*pi*4*t);

A = A_sig(time);
B = B_sig(time);
C = C_sig(time);
D = D_sig(time);

names = {'a1', 'a2', 'b1', 'b2'};
% save A with differnt names

base_path = fullfile('.');

for e = 1:2
    for rep = 1:2
        for m = 1:length(names)
            my_struct = struct();
            save_filename = fullfile(base_path, sprintf('E%d_set0%d_rep0%d_%s.mat', e, e, rep, names{m}));

            if m == 1
                my_struct(1).name = 'm_1';
                my_struct(1).time = A;

                my_struct(2).name = 'm_2';
                my_struct(2).time = B;

            elseif m == 2
                my_struct(1).name = 'n_1';
                my_struct(1).time = A;

                my_struct(2).name = 'n_2';
                my_struct(2).time = B;

            elseif m == 3
                my_struct(1).name = 't_1';
                my_struct(1).time = A;
                
                my_struct(2).name = 't_2';
                my_struct(2).time = B;

                my_struct(3).name = 't_3';
                my_struct(3).time = C;

                my_struct(4).name = 't_4';
                my_struct(4).time = D;

            elseif m == 4
                my_struct(1).name = 'r_1';
                my_struct(1).time = A;
                
                my_struct(2).name = 'r_2';
                my_struct(2).time = B;

                my_struct(3).name = 'r_3';
                my_struct(3).time = C;

                my_struct(4).name = 'r_4';
                my_struct(4).time = D;

            end
                % Save struct
                save(save_filename, 'my_struct');
        end
    end
end

% S = struct();
% save_filename = fullfile(base_path, 'b2.mat');
% varname_node1 = 'r_1';
% varname_node2 = 'r_2';
% S.(varname_node1) = A;
% S.(varname_node2) = B;
% save(save_filename, '-struct', 'S');

% % Create struct
% my_struct(1).name = 'm1';
% my_struct(1).time = A;
% 
% my_struct(2).name = 'm2';
% my_struct(2).time = B;
% 
% % Save struct
% save('myData.mat', 'my_struct');
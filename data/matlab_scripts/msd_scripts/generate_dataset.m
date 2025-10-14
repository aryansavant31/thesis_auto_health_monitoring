function adj = generate_dataset(machine_type, scenario, ds_type, ds_subtype, ds_num)

    % Get dynamics outputs
    [pos, vel, acc, time, conn_pairs] = run_dynamics(machine_type, scenario, ds_type, ds_subtype, ds_num);

    % Signal types and corresponding data
    signal_types = {'pos', 'vel', 'acc'};
    signals = {pos, vel, acc};

    % Number of masses (nodes)
    M = size(pos, 2);

    % Base path to save
    ds_subtype_str = ds_subtype + ds_num;
    base_path = fullfile('.', 'data', 'datasets', 'mass_sp_dm', machine_type, scenario, ds_type, ds_subtype_str, 'raw');

    % Save signals (per mass, per signal type)
    for s = 1:length(signal_types)
        signal_type = signal_types{s};
        signal_data = signals{s};  % Each is [timesteps x M]

        for m = 1:M
            S = struct();
            % node_data = signal_data(:, m);  % Time series for mass m

            save_path = fullfile(base_path, 'nodes', sprintf('mass_%d', m), signal_type);
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end

            % Save as <ds_subtype>.mat
            save_filename = fullfile(save_path, strcat(signal_type, '_node', '.mat'));

            % varname_node_data = sprintf('%s_mass_%d_data', signal_type, m);
            % varname_node_time = sprintf('%time_mass_%d', m);
            S.('name') = sprintf('%s_mass_%d', signal_type, m);
            S.('data') = signal_data(:, m); % Time series for mass m
            S.('time') = time;

            save(save_filename, 'S');
        end
    end

    % ============== Create and save adjacency matrix ==============
    adj = zeros(M, M);  % Preallocate MxM matrix

    for i = 1:size(conn_pairs, 1)
        from = conn_pairs(i, 1);
        to = conn_pairs(i, 2);
        adj(from, to) = 1;
        adj(to, from) = 1;  % Undirected
    end

    % Save adjacency matrix
    edge_save_path = fullfile(base_path, 'edges');
    if ~exist(edge_save_path, 'dir')
        mkdir(edge_save_path);
    end
    adj_filename = fullfile(edge_save_path, strcat(ds_subtype_str, '_adj.mat'));
    
    E = struct();
    varname_edge = sprintf('%s_adj', ds_subtype);
    E.(varname_edge) = adj; 
            
    save(adj_filename, '-struct', 'E');

end
    




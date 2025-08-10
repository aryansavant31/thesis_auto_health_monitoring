function generate_dataset(machine_type, scenario, ds_type, ds_subtype)

    % Get dynamics outputs
    [pos, vel, acc, conn_pairs] = run_dynamics(machine_type, scenario, ds_type, ds_subtype);

    % Signal types and corresponding data
    signal_types = {'pos', 'vel', 'acc'};
    signals = {pos, vel, acc};

    % Number of masses (nodes)
    M = size(pos, 2);

    % Base path to save
    base_path = fullfile('.', 'data', 'datasets', 'mass_sp_dm', machine_type, scenario, ds_type, ds_subtype, 'raw');

    % Save signals (per mass, per signal type)
    for s = 1:length(signal_types)
        signal_type = signal_types{s};
        signal_data = signals{s};  % Each is [timesteps x M]

        for m = 1:M
            S = struct();
            % node_data = signal_data(:, m);  % Time series for mass m

            save_path = fullfile(base_path, 'nodes', sprintf('%d_mass_%d', m, m), signal_type);
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end

            % Save as <ds_subtype>.mat
            save_filename = fullfile(save_path, strcat(ds_subtype, '_node', '.mat'));

            varname_node = sprintf('%s_mass_%d_time', signal_type, m);
            S.(varname_node) = signal_data(:, m); % Time series for mass m

            save(save_filename, '-struct', 'S');
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
    adj_filename = fullfile(edge_save_path, strcat(ds_subtype, '_adj.mat'));
    
    E = struct();
    varname_edge = sprintf('%s_adj', ds_subtype);
    E.(varname_edge) = adj; 
            
    save(adj_filename, '-struct', 'E');

end
    




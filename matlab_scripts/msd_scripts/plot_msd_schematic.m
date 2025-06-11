function plot_msd_schematic(M, nl_index)
    figure; hold on; axis equal;
    x_spacing = 3;
    y_level = 0;
    
    block_w = 1; block_h = 1;
    
    % Wall position
    wall_x = 0;
    
    for i = 1:M
        % Position of current mass block
        x_block = wall_x + i * x_spacing;
        
        % Draw mass block
        rectangle('Position', [x_block - block_w/2, y_level, block_w, block_h], ...
                  'FaceColor', [0.7 0.8 1], 'EdgeColor', 'k', 'LineWidth', 2);
        text(x_block, y_level + block_h + 0.3, sprintf('m_%d', i), ...
             'HorizontalAlignment', 'center', 'FontSize', 12);
        
        % Draw wall connections if specified
        if ismember(i, nl_index.springs_wall)
            plot_spring(wall_x, x_block - block_w/2, y_level + block_h/2, 'left');
        end
        if ismember(i, nl_index.dampers_wall)
            plot_damper(wall_x, x_block - block_w/2, y_level - 1, 'left');
        end
        
        % Connect to previous mass
        if i > 1
            x_prev = wall_x + (i - 1) * x_spacing;
            if ismember(i - 1, nl_index.springs)
                plot_spring(x_prev + block_w/2, x_block - block_w/2, y_level + block_h/2, 'center');
            end
            if ismember(i - 1, nl_index.dampers)
                plot_damper(x_prev + block_w/2, x_block - block_w/2, y_level - 1, 'center');
            end
        end
    end

    % Draw input force on last block
    x_input = wall_x + M * x_spacing;
    arrow_length = 1.5;
    quiver(x_input + block_w/2 + 0.1, y_level + block_h/2, arrow_length, 0, ...
           'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
    text(x_input + block_w/2 + arrow_length + 0.1, y_level + block_h/2, ...
         'Input u(t)', 'Color', 'r', 'FontSize', 12, ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');

    title('Mass-Spring-Damper Schematic');
    axis off;
end

function plot_spring(x1, x2, y, type)
    plot([x1 x2], [y y], 'Color', [0 0.6 0], 'LineWidth', 2); % Simplified spring
    if strcmp(type, 'left')
        text((x1+x2)/2, y + 0.2, 'k_{wall}', 'HorizontalAlignment', 'center');
    else
        text((x1+x2)/2, y + 0.2, 'k', 'HorizontalAlignment', 'center');
    end
end

function plot_damper(x1, x2, y, type)
    plot([x1 x2], [y y], 'Color', [0 0 1], 'LineWidth', 2); % Simplified damper
    if strcmp(type, 'left')
        text((x1+x2)/2, y - 0.3, 'd_{wall}', 'HorizontalAlignment', 'center');
    else
        text((x1+x2)/2, y - 0.3, 'd', 'HorizontalAlignment', 'center');
    end
end
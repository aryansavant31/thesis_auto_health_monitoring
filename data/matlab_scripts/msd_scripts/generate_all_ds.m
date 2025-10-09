function generate_all_ds

    for i = 1:12
        str = sprintf("ds_%d", i);
        generate_dataset("M005", "scene_1", "healthy", str)
    end
    
    for i = 1:10
        for j = 1:3
            str = sprintf("top_add_%d_ds_%d", i, j);
            generate_dataset("M005", "scene_1", "unhealthy", str)
        end
        
    end
end
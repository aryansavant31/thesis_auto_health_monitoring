import os, sys

# other imports
import argparse
import concurrent.futures
import numpy as np
from collections import defaultdict
from pathlib import Path

# global imports
from data.config import DataSweep
from console_logger import ConsoleLogger

# local imports
from .settings.manager import FaultDetectorInferSweepManager
from .infer import FaultDetectorInferPipeline
from .logs.mass_sp_dm.M005.scene_1.train.feature_chart_maker import plot_feature_importance_from_excel

def make_fdet_results_dict(preds, model_name, results_dict):
    model_id = model_name.split('-')[0]

    node_group = model_id.split('(')[0]
    signal_group = model_id.split('(')[1].split('+')[0]
    set_id = model_id.split('+')[1].strip(')')

    results_dict[set_id][node_group][signal_group] = preds

    return results_dict

def infer_single_config(args):
    """
    Infer a single configuration.

    Parameters
    ----------
    args : tuple
        (idx, fdet_config, total_configs)
    """
    idx, fdet_config, total_configs, run_type = args

    console_logger_infer = ConsoleLogger()

    print(f"\nConfiguration {idx+1}/{total_configs}")

    with console_logger_infer.capture_output():
        print(f"\nStarting anomaly detector to {run_type}...")

        infer_pipeline = FaultDetectorInferPipeline(fdet_config.data_config, fdet_config)
        if run_type == 'custom_test':
            infer_pipeline.infer()

        elif run_type == 'predict':
            preds = infer_pipeline.infer()
        
        base_name = f"{fdet_config.selected_model_num}/{os.path.basename(infer_pipeline.infer_log_path)}" if infer_pipeline.infer_log_path else f"{fdet_config.selected_model_num}/{run_type}"
        print('\n' + 75*'=')
        print(f"\n{run_type.capitalize()} of anomaly detector '{base_name}' completed.")

    if fdet_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(infer_pipeline.infer_log_path, "console_output.txt")
        console_logger_infer.save_to_file(file_path, script_name="fault_detection.infer.py", base_name=base_name)

    print('\n' + 75*'%')

    return {
        'idx': idx,
        'base_name': base_name,
        'model_name': fdet_config.selected_model_num,
        'preds': preds if run_type == 'predict' else None,
        'data_config': fdet_config.data_config
    }

def multi_inferer_main(run_type, parallel_execution, max_workers=None):
    """
    Main function to handle multiple inference configurations.

    Parameters
    ----------
    run_type : str
        The type of inference run ('custom_test' or 'predict').
    parallel_execution : bool
        Whether to run in parallel mode or sequentially.
    max_workers : int, optional
        Maximum number of workers. If None, uses default (number of processors)
    """
    if max_workers is None:
        max_workers = os.cpu_count()

    console_logger_sweep = ConsoleLogger()

    with console_logger_sweep.capture_output():
        print("\nStarting fault detection model inference sweep...")

        # get data configuration
        data_sweep = DataSweep(run_type=run_type)
        data_configs = data_sweep.get_sweep_configs()

        print(f"\nProcessing {len(data_configs)} data groups")
        data_sweep.print_sweep_summary()

        # get all inference configurations
        sweep_manager = FaultDetectorInferSweepManager(data_configs=data_configs, run_type=run_type)
        fdet_configs = sweep_manager.get_sweep_configs()
        sweep_manager.print_sweep_summary()

        infer_sweep_num = sweep_manager.infer_sweep_num

        fdet_results = {'module':[], 'signal_type':[], 'score':[], 'pred_label':[]} # to store results of all the models

        config_args = [(idx, fdet_config, len(fdet_configs), run_type) 
                              for idx, fdet_config in enumerate(fdet_configs)]

        if parallel_execution:
            print(f"\nParallelizing {len(fdet_configs)} fault detection configurations across {max_workers} workers...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all configuration training tasks
                future_to_config = {
                    executor.submit(infer_single_config, arg): arg 
                    for arg in config_args
                }

                for future in concurrent.futures.as_completed(future_to_config):
                    try:
                        result = future.result()
                        fdet_results['module'].append(result['data_config'].signal_types['node_group_name'])
                        fdet_results['signal_type'].append(result['data_config'].signal_types['signal_group_name'])
                        fdet_results['score'].append(float(result['preds']['sign_scores'].iloc[0]))
                        fdet_results['pred_label'].append(float(result['preds']['final_pred_label'].iloc[0]))
                        # if result['preds'] is not None:
                        #     results_dict = make_fdet_results_dict(result['preds'], result['model_name'], results_dict)
                        print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")
                    
                    except Exception as e:
                        config_arg = future_to_config[future]
                        print(f"  Config {config_arg[0] + 1} failed with error: {e}")

                # Sort results by index to maintain order
                # config_results.sort(key=lambda x: x['idx'])

        else:
            print(f"\nRunning {len(fdet_configs)} fault detection inference configurations sequentially...")

            results_dict = defaultdict(lambda: defaultdict(dict))
            for arg in config_args:
                result = infer_single_config(arg)
                fdet_results['module'].append(result['data_config'].signal_types['node_group_name'])
                fdet_results['signal_type'].append(result['data_config'].signal_types['signal_group_name'])
                fdet_results['score'].append(float(result['preds']['sign_scores'].iloc[0]))
                fdet_results['pred_label'].append(float(result['preds']['final_pred_label'].iloc[0]))
                # if result['preds'] is not None:
                #     results_dict = make_fdet_results_dict(result['preds'], result['model_name'], results_dict)
                print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")

        print('\n' + 75*'=')
        print('\n' + 75*'=')

        print("\n")
        print(fdet_results)
        diagnostic_dashboard(fdet_results)

        fdet_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_file_path = os.path.join(fdet_dir, "docs", "sweep_logs", f"{run_type}_sweeps")
        if not os.path.exists(sweep_file_path):
            os.makedirs(sweep_file_path)

        # if run_type == 'predict':
        #     npy_file_path = os.path.join(sweep_file_path, f"fdet_results_dict.npy")
        #     np.save(npy_file_path, results_dict)
        #     print(f"\nFault detection results dictionary saved at: {npy_file_path}")
        
        text_file_path = os.path.join(sweep_file_path, f"iswp_{infer_sweep_num}_output.txt")
        print(f"\nFault detection model '{run_type}' sweep completed. Sweep log saved at: {text_file_path}")

    console_logger_sweep.save_to_file(text_file_path, script_name="fault_detection.multi_infer.py", base_name=f"iswp_{infer_sweep_num}")

def diagnostic_dashboard(results):
    """
    Generate a diagnostic dashboard based on the results.

    Parameters
    ----------
    results : dict
        The results dictionary containing predictions and scores.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Color mapping for pred_label
    color_map = {
        -1: '#FF6666',      # red
        -0.5: '#FF9933',    # light red
        0.5: '#B3FF66',     # light green
        1: '#59C459',       # green
    }
    theme_map = {
        -1: 'red',
        -0.5: 'light_red',
        0.5: 'light_green',
        1: 'green'
    }

    # List of feature plot specs
    plot_specs = [
        # mass 1 - acc
        {'module': 'mass_1', 'signal_type': 'acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_1\acc\set_G\IF\tswp_1\[mass_1_(acc+G)]-IF_fdet_64\feature_sheet.xlsx")},
        # mass 1 - pos
        {'module': 'mass_1', 'signal_type': 'pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_1\pos\set_G\IF\tswp_1\[mass_1_(pos+G)]-IF_fdet_7\feature_sheet.xlsx")},
        # mass 1 - vel
        {'module': 'mass_1', 'signal_type': 'vel', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_1\vel\set_G\IF\tswp_1\[mass_1_(vel+G)]-IF_fdet_55\feature_sheet.xlsx")},
        # mass 2 - acc
        {'module': 'mass_2', 'signal_type': 'acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_2\acc\set_G\IF\tswp_1\[mass_2_(acc+G)]-IF_fdet_45\feature_sheet.xlsx")},
        # mass 2 - pos
        {'module': 'mass_2', 'signal_type': 'pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_2\pos\set_G\IF\tswp_1\[mass_2_(pos+G)]-IF_fdet_4\feature_sheet.xlsx")},
        # mass 2 - vel
        {'module': 'mass_2', 'signal_type': 'vel', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_2\vel\set_G\IF\tswp_1\[mass_2_(vel+G)]-IF_fdet_29\feature_sheet.xlsx")},
        # mass 3 - acc
        {'module': 'mass_3', 'signal_type': 'acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_3\acc\set_G\IF\tswp_1\[mass_3_(acc+G)]-IF_fdet_47\feature_sheet.xlsx")},
        # mass 3 - pos
        {'module': 'mass_3', 'signal_type': 'pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_3\pos\set_G\IF\tswp_1\[mass_3_(pos+G)]-IF_fdet_30\feature_sheet.xlsx")},
        # mass 3 - vel
        {'module': 'mass_3', 'signal_type': 'vel', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_3\vel\set_G\IF\tswp_1\[mass_3_(vel+G)]-IF_fdet_55\feature_sheet.xlsx")},
        # mass 4 - acc
        {'module': 'mass_4', 'signal_type': 'acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_4\acc\set_G\IF\tswp_1\[mass_4_(acc+G)]-IF_fdet_42\feature_sheet.xlsx")},
        # mass 4 - pos
        {'module': 'mass_4', 'signal_type': 'pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_4\pos\set_G\IF\tswp_1\[mass_4_(pos+G)]-IF_fdet_35\feature_sheet.xlsx")},
        # mass 4 - vel
        {'module': 'mass_4', 'signal_type': 'vel', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_4\vel\set_G\IF\tswp_1\[mass_4_(vel+G)]-IF_fdet_40\feature_sheet.xlsx")},
        # mass 5 - acc
        {'module': 'mass_5', 'signal_type': 'acc', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_5\acc\set_G\IF\tswp_1\[mass_5_(acc+G)]-IF_fdet_38\feature_sheet.xlsx")},
        # mass 5 - pos
        {'module': 'mass_5', 'signal_type': 'pos', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_5\pos\set_G\IF\tswp_1\[mass_5_(pos+G)]-IF_fdet_7\feature_sheet.xlsx")},
        # mass 5 - vel
        {'module': 'mass_5', 'signal_type': 'vel', 'excel_path': Path(r"C:\Aryan_Savant\Thesis_Projects\my_work\AFD_thesis\fault_detection\logs\mass_sp_dm\M005\scene_1\train\mass_5\vel\set_G\IF\tswp_1\[mass_5_(vel+G)]-IF_fdet_72\feature_sheet.xlsx")},
    ]

    # Desired signal type order
    signal_type_order = ['acc', 'pos', 'vel']

    modules = results['module']
    signal_types = results['signal_type']
    scores = results['score']
    pred_labels = results['pred_label']

    # Get unique modules and their indices
    unique_modules = list(sorted(set(modules)))
    module_to_indices = {mod: [i for i, m in enumerate(modules) if m == mod] for mod in unique_modules}

    n_modules = len(unique_modules)
    n_cols = 3
    n_rows = int(np.ceil(n_modules / n_cols))

    fig_width = 6 * n_cols
    fig_height = 15 * n_rows  # slightly reduced for compactness
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, module in enumerate(unique_modules):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.set_title(f"Module: {module}", fontsize=13, fontweight='bold', pad=-15)
        ax.axis('off')

        ax.text(0.08, 0.98, f"Module: {module}", fontsize=13, fontweight='bold', 
                ha='left', va='top', transform=ax.transAxes)

        indices = module_to_indices[module]
        module_signal_types = [signal_types[i] for i in indices]
        module_scores = [scores[i] for i in indices]
        module_pred_labels = [pred_labels[i] for i in indices]

        # Sort by desired signal type order
        zipped = list(zip(module_signal_types, module_scores, module_pred_labels))
        zipped_sorted = sorted(
            zipped,
            key=lambda x: signal_type_order.index(x[0]) if x[0] in signal_type_order else len(signal_type_order)
        )
        module_signal_types, module_scores, module_pred_labels = zip(*zipped_sorted) if zipped_sorted else ([], [], [])

        n_signals = len(module_signal_types)
        box_w = 0.75  # slightly smaller
        box_h = 0.22  # fixed small height for each signal type box
        x = 0.08
        spacing_multiplier = 1.1

         # Calculate starting y position based on number of signals
        total_height = (n_signals - 1) * box_h * spacing_multiplier + box_h
        y_start = 0.9 - total_height

        for j, (stype, score, plabel) in enumerate(zip(module_signal_types, module_scores, module_pred_labels)):
            y = y_start + (n_signals - 1 - j) * box_h * spacing_multiplier
            # Get color for pred_label (handle array or scalar)
            if isinstance(plabel, (list, np.ndarray)):
                val = plabel[0] if len(plabel) > 0 else 0
            else:
                val = plabel
            color = color_map.get(val, '#CCCCCC')
            theme = theme_map.get(val, 'green')

            # Get score as text (handle array or scalar)
            if isinstance(score, (list, np.ndarray)):
                score_val = score[0] if len(score) > 0 else ''
            else:
                score_val = score

            rect = patches.FancyBboxPatch(
                (x, y), box_w, box_h*0.98, boxstyle="round,pad=0.01", linewidth=1, edgecolor='black', facecolor=color
            )
            ax.add_patch(rect)
            ax.text(x + box_w/2, y + box_h*0.88, f"{stype}", ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.text(x + box_w/2, y + box_h*0.75, f"Score: {score_val:.3f}", ha='center', va='top', fontsize=11)

            # Find the correct excel_path for this module and signal_type
            excel_path = None
            for spec in plot_specs:
                if spec['module'] == module and spec['signal_type'] == stype:
                    excel_path = spec['excel_path']
                    break

            # Attach feature plot below the score value
            if excel_path and excel_path.exists():
                try:
                    feature_fig, feature_axes = plot_feature_importance_from_excel(excel_path=excel_path, theme=theme, bg_alpha=1)
                    
                    # Resize the feature figure to be smaller for embedding
                    feature_fig.set_size_inches(12, 6)
                    feature_fig.tight_layout()
                    
                    # Render the entire feature figure to an image
                    feature_fig.canvas.draw()
                    image = np.frombuffer(feature_fig.canvas.buffer_rgba(), dtype=np.uint8)
                    image = image.reshape(feature_fig.canvas.get_width_height()[::-1] + (4,))
                    # Convert RGBA to RGB by dropping the alpha channel
                    image = image[:, :, :3]
                    
                    # Create inset axis for the feature plot with better positioning
                    # Position it below the score text, within the box
                    ax_inset = ax.inset_axes([x + 0.01, y + 0.01, box_w, box_h])
                    
                    # Display the image in the inset
                    ax_inset.imshow(image)
                    ax_inset.axis('off')
                    
                    plt.close(feature_fig)
                except Exception as e:
                    print(f"Warning: Could not render feature plot for {module}/{stype}: {e}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for idx in range(n_modules, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer multiple anomaly detector models based on sweep configurations.")

    parser.add_argument('--run-type', type=str, 
                    choices=['custom_test', 'predict'],
                    default='predict',
                    required=True, help="Run type: custom_test or predict")
    
    parser.add_argument('--parallel', action='store_true',
                    help="Enable parallel execution of inference configurations.")
    
    parser.add_argument('--max-workers', type=int, default=8,
                    help="Maximum number of workers for parallel execution. Default is 8.")
    
    args = parser.parse_args()

    multi_inferer_main(run_type=args.run_type, 
                        parallel_execution=args.parallel, 
                        max_workers=args.max_workers)

        

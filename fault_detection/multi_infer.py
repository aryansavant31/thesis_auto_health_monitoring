import os, sys

# other imports
import argparse
import concurrent.futures
import numpy as np
from collections import defaultdict

# global imports
from data.config import DataSweep
from console_logger import ConsoleLogger

# local imports
from .settings.manager import AnomalyDetectorInferSweepManager
from .infer import AnomalyDetectorInferPipeline

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

        infer_pipeline = AnomalyDetectorInferPipeline(fdet_config.data_config, fdet_config)
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
        'preds': preds if run_type == 'predict' else None
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
        fdet_sweep = AnomalyDetectorInferSweepManager(data_configs=data_configs, run_type=run_type)
        fdet_configs = fdet_sweep.get_sweep_configs()
        fdet_sweep.print_sweep_summary()

        infer_sweep_num = fdet_sweep.infer_sweep_num

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

                # Collect results as they complete
                config_results = []
                results_dict = defaultdict(lambda: defaultdict(dict))

                for future in concurrent.futures.as_completed(future_to_config):
                    try:
                        result = future.result()
                        config_results.append(result)
                        if result['preds'] is not None:
                            results_dict = make_fdet_results_dict(result['preds'], result['model_name'], results_dict)
                        print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")
                    
                    except Exception as e:
                        config_arg = future_to_config[future]
                        print(f"  Config {config_arg[0] + 1} failed with error: {e}")

                # Sort results by index to maintain order
                config_results.sort(key=lambda x: x['idx'])

        else:
            print(f"\nRunning {len(fdet_configs)} fault detection inference configurations sequentially...")

            results_dict = defaultdict(lambda: defaultdict(dict))
            for arg in config_args:
                result = infer_single_config(arg)
                if result['preds'] is not None:
                    results_dict = make_fdet_results_dict(result['preds'], result['model_name'], results_dict)
                print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")

        print('\n' + 75*'=')
        print('\n' + 75*'=')

        fdet_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_file_path = os.path.join(fdet_dir, "docs", "sweep_logs", f"{run_type}_sweeps")
        if not os.path.exists(sweep_file_path):
            os.makedirs(sweep_file_path)

        if run_type == 'predict':
            npy_file_path = os.path.join(sweep_file_path, f"fdet_results_dict.npy")
            np.save(npy_file_path, results_dict)
            print(f"\nFault detection results dictionary saved at: {npy_file_path}")
        
        text_file_path = os.path.join(sweep_file_path, f"iswp_{infer_sweep_num}_output.txt")
        print(f"\nFault detection model '{run_type}' sweep completed. Sweep log saved at: {text_file_path}")

    console_logger_sweep.save_to_file(text_file_path, script_name="fault_detection.multi_infer.py", base_name=f"iswp_{infer_sweep_num}")

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

        

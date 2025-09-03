import os, sys
import concurrent.futures
from functools import partial
import argparse

# global imports
from data.config import DataSweep
from console_logger import ConsoleLogger

# local imports
from .settings.manager import NRITrainSweepManager, DecoderTrainSweepManager
from .train import NRITrainPipeline, DecoderTrainPipeline

def train_single_config(args):
    """
    Train a single configuration. Used for parallel processing of loop B.
    
    Parameters
    ----------
    args : tuple
        (idx, tp_config, total_configs, framework, device)
    """
    idx, tp_config, total_configs, framework, device = args

    console_logger_train = ConsoleLogger()

    print(f"\nConfiguration {idx+1}/{total_configs}")
    print("\n Training Parameters:")
    for key, value in tp_config.hyperparams.items():
        print(f"  {key}: {value}")

    # run the training pipeline for the current config
    with console_logger_train.capture_output():
        print(f"\nStarting {framework} model training...")

        if framework == 'nri':
            train_pipeline = NRITrainPipeline(tp_config.data_config, tp_config)
        elif framework == 'decoder':
            train_pipeline = DecoderTrainPipeline(tp_config.data_config, tp_config)
        else:
            raise ValueError(f"Invalid framework: {framework}. Choose 'nri' or 'decoder'.")
        
        train_pipeline.train(device=device)
        
        base_name = os.path.basename(train_pipeline.train_log_path) if train_pipeline.train_log_path else f"{framework}_model"
        print('\n' + 75*'=')
        print(f"\n{framework.capitalize()} model '{base_name}' training completed.")

    if train_pipeline.train_log_path:
        # save the captured output to a file
        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
        console_logger_train.save_to_file(file_path, script_name="topology_estimation.train.py", base_name=base_name)

    print('\n' + 75*'%')
    
    return {
        'idx': idx,
        'config': tp_config,
        'train_log_path': train_pipeline.train_log_path,
        'base_name': base_name
    }

def multi_train(framework, parallel_execution, max_workers=None):
    """
    Main function to run multiple training sessions for the anomaly detector.
    
    Parameters
    ----------
    parallel_mode : bool
        Whether to run in parallel mode or sequentially.

    max_workers : int, optional
        Maximum number of workers. If None, uses default (number of processors)
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    console_logger_sweep = ConsoleLogger()
    
    with console_logger_sweep.capture_output():
        print(f"\nStarting {framework} model train sweep...")

        # get all data configuration groups
        data_sweep = DataSweep(run_type='train')
        data_configs = data_sweep.get_sweep_configs()

        print(f"\nProcessing {len(data_configs)} data groups")
        data_sweep.print_sweep_summary()

        # get all training configurations (of NRI or Decoder)
        if framework == 'nri':
            sweep_manager = NRITrainSweepManager(data_configs)
        elif framework == 'decoder':
            sweep_manager = DecoderTrainSweepManager(data_configs)
        else:
            raise ValueError(f"Invalid framework: {framework}. Choose 'nri' or 'decoder'.")
        
        tp_configs = sweep_manager.get_sweep_configs()
        sweep_manager.print_sweep_summary()

        train_sweep_num = sweep_manager.train_sweep_num

        if parallel_execution:
            print(f"Parallelizing {len(tp_configs)} {framework} configurations across {max_workers} workers...")
            
            device = 'cpu'
            print(f"\nSince running configs sequenctially, using device: {device}")

            config_args = [(idx, tp_config, len(tp_configs), framework, device) 
                           for idx, tp_config in enumerate(tp_configs)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all configuration training tasks
                future_to_config = {
                    executor.submit(train_single_config, arg): arg 
                    for arg in config_args
                }
                    
                # Collect results as they complete
                config_results = []
                for future in concurrent.futures.as_completed(future_to_config):
                    try:
                        result = future.result()
                        config_results.append(result)
                        print(f"  Completed {framework} config {result['idx'] + 1}: {result['base_name']}")
                    except Exception as e:
                        config_arg = future_to_config[future]
                        print(f"  {framework.capitalize()} config {config_arg[0] + 1} failed with error: {e}")
                
                # Sort results by index to maintain order
                config_results.sort(key=lambda x: x['idx'])

        else:
            print(f"\nRunning {len(tp_configs)} {framework} configurations sequentially (no parallelization)...")
            
            device = 'auto'
            print(f"\nSince running configs sequenctially, using device: {device}")

            config_args = [(idx, tp_config, len(tp_configs), framework, device) 
                           for idx, tp_config in enumerate(tp_configs)]
            
            for arg in config_args:
                result = train_single_config(arg)
                print(f"  Completed {framework} config {result['idx'] + 1}: {result['base_name']}")

        
        print('\n' + 75*'=')
        print('\n' + 75*'=')

        tp_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_file_path = os.path.join(tp_dir, "docs", "sweep_logs", framework, "train_sweeps") 
        if not os.path.exists(sweep_file_path):
            os.makedirs(sweep_file_path)
            
        text_file_path = os.path.join(sweep_file_path, f"tswp_{train_sweep_num}_console_output.txt")
        print(f"\n{framework.capitalize()} model sweep completed. Sweep log saved at: {text_file_path}")
        
    console_logger_sweep.save_to_file(text_file_path, script_name="topology_estimation.multi_train.py", base_name=f"tswp_{train_sweep_num}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple NRI or decoder models for topology estimation.")

    parser.add_argument('--framework', type=str,
                        choices=['nri', 'decoder'],
                        default='nri',
                        required=True, help="Framework to use: 'nri' or 'decoder'")
    
    parser.add_argument('--parallel', action='store_true',
                        help="Enable parallel execution of training configurations.")
    
    parser.add_argument('--max-workers', type=int, default=8,
                        help="Maximum number of workers for parallel execution. Default is 8.")
    
    args = parser.parse_args()

    multi_train(framework=args.framework,
                parallel_execution=args.parallel,
                max_workers=args.max_workers
                )
    
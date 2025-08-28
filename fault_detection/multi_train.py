import os, sys
import concurrent.futures
from functools import partial

# global imports
from data.config import DataConfig
from data.datasets.bearing.groups import BERGroupMaker
from console_logger import ConsoleLogger

# local imports
from .settings.manager import AnomalyDetectorSweepManager
from .single_train import AnomalyDetectorTrainPipeline

class MultiTrainerConfig:
    def __init__(self):
        """
        Attributes
        ----------
        parallel_mode : str
            It defines whether to parallelize the model training on a dataset level or configuration level. 
            Dataset level is above configuration level.
            Options:
            'none' - no parallelization (default)
            'data_level' - parallelize on dataset level
            'fdet_config_level' - parallelize on model configuration level

        max_workers : int
            Maximum number of workers for parallel processing.
            If None, uses the default (number of processors).

        data_config_groups : list
            List of data configuration groups to process. Each group is a list containing:
            [`signal_types`, `set_id`,]
        """
        self.parallel_mode = 'fdet_config_level' 
        self.max_workers = 12
        self.data_config_groups = [
            [BERGroupMaker().gb_acc, 'G']
        ]


def train_single_config(args):
    """
    Train a single configuration. Used for parallel processing of loop B.
    
    Parameters
    ----------
    args : tuple
        (idx, fdet_config, data_config, total_configs)
    """
    idx, fdet_config, data_config, total_configs = args
    
    console_logger_train = ConsoleLogger()
    
    print(f"\nConfiguration {idx+1}/{total_configs}")
    print("\n Training Parameters:")
    for key, value in fdet_config.hparams.items():
        print(f"  {key}: {value}")
    
    # run the training pipeline for the current config
    with console_logger_train.capture_output():
        print("\nStarting fault detection model training...")

        train_pipeline = AnomalyDetectorTrainPipeline(data_config, fdet_config)
        train_pipeline.train()

        base_name = os.path.basename(train_pipeline.train_log_path) if train_pipeline.train_log_path else fdet_config.anom_config['anom_type']
        print('\n' + 75*'=')
        print(f"\nFault detection model '{base_name}' training completed.")

    if fdet_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
        console_logger_train.save_to_file(file_path, script_name="fault_detection.train.py", base_name=base_name)

    print('\n' + 75*'%')
    
    return {
        'idx': idx,
        'config': fdet_config,
        'train_log_path': train_pipeline.train_log_path,
        'base_name': base_name
    }

def process_single_data_group(args):
    """
    Process a single data group. Used for parallel processing on dataset level.
    
    Parameters
    ----------
    args : tuple
        (group_idx, group, sweep_num)
    """
    group_idx, group = args
    
    results = []
    
    print(f"\nProcessing group {group_idx + 1}")
    
    data_config = DataConfig(run_type="train")
    data_config.signal_types = group[0]
    data_config.set_id = group[1]

    sweep_config = AnomalyDetectorSweepManager(data_config, make_model_num=False)
    
    print("\nStarting fault detection model sweep for this group...")
    sweep_config.print_sweep_summary()

    # get all training configurations
    fdet_configs = sweep_config.get_sweep_configs()

    # train models for each configuration (sequential within group)
    for idx, fdet_config in enumerate(fdet_configs):
        result = train_single_config((idx, fdet_config, data_config, len(fdet_configs)))
        results.append(result)
    
    return {
        'group_idx': group_idx,
        'group': group,
        'results': results,
        'sweep_num': sweep_config.sweep_num
    }

def multi_trainer_main(data_config_groups, parallel_mode='none', max_workers=None):
    """
    Main function to run multiple training sessions for the anomaly detector.
    
    Parameters
    ----------
    parallel_mode : str

    max_workers : int, optional
        Maximum number of workers. If None, uses default (number of processors)
    """
    if max_workers is None:
        max_workers = os.cpu_count()
    
    if parallel_mode not in ['none', 'data_level', 'fdet_config_level']:
        raise ValueError("parallel_mode must be 'none', 'data_level' or 'model_config_level'")
    
    print(f"\nRunning with parallel_mode='{parallel_mode}' using {max_workers} workers")
    
    # Parameters
    console_logger_sweep = ConsoleLogger()
    
    
    with console_logger_sweep.capture_output():
        print("\nStarting fault detection model sweep...")
        
        if parallel_mode == 'data_level':
            # Parallelize data config loop (loop A) (groups)
            print(f"Parallelizing data groups across {max_workers} workers...")
            
            # Prepare arguments for parallel processing
            group_args = [(i, group) for i, group in enumerate(data_config_groups)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all group processing tasks
                future_to_group = {
                    executor.submit(process_single_data_group, arg): arg 
                    for arg in group_args
                }
                
                # Collect results as they complete
                group_results = []
                for future in concurrent.futures.as_completed(future_to_group):
                    try:
                        result = future.result()
                        sweep_num = result['sweep_num']
                        group_results.append(result)
                        print(f"\nCompleted group {result['group_idx'] + 1}")
                        for res in result['results']:
                            print(f"  Config {res['idx'] + 1}: {res['base_name']}")
                    except Exception as e:
                        group_arg = future_to_group[future]
                        print(f"\nGroup {group_arg[0] + 1} failed with error: {e}")
        
        elif parallel_mode == 'fdet_config_level':
            # Parallelize hyperparameter sweep (loop B (configurations)) - groups are processed sequentially
            print(f"Parallelizing fault detector configurations across {max_workers} workers...")
            
            for group_idx, group in enumerate(data_config_groups):
                print(f"\nProcessing group {group_idx + 1}/{len(data_config_groups)}")
                
                data_config = DataConfig(run_type="train")
                data_config.signal_types = group[0]
                data_config.set_id = group[1]

                sweep_config = AnomalyDetectorSweepManager(data_config, make_model_num=True)
                sweep_num = sweep_config.sweep_num

                sweep_config.print_sweep_summary()

                # get all training configurations
                fdet_configs = sweep_config.get_sweep_configs()
                
                print(f"Parallelizing {len(fdet_configs)} configurations across {max_workers} workers...")
                
                # Prepare arguments for parallel processing
                config_args = [(idx, fdet_config, data_config, len(fdet_configs)) 
                              for idx, fdet_config in enumerate(fdet_configs)]
                
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
                            print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")
                        except Exception as e:
                            config_arg = future_to_config[future]
                            print(f"  Config {config_arg[0] + 1} failed with error: {e}")
                    
                    # Sort results by index to maintain order
                    config_results.sort(key=lambda x: x['idx'])
        
        else:
            # No parallelization - original sequential code
            print("Running sequentially (no parallelization)...")
            
            for group in data_config_groups:
                data_config = DataConfig(run_type="train")
                data_config.signal_types = group[0]
                data_config.set_id = group[1]

                sweep_config = AnomalyDetectorSweepManager(data_config, make_model_num=True)
                sweep_num = sweep_config.sweep_num

                sweep_config.print_sweep_summary()

                # get all training configurations
                fdet_configs = sweep_config.get_sweep_configs()

                # train models for each configuration
                for idx, fdet_config in enumerate(fdet_configs):
                    console_logger_train = ConsoleLogger()
                    
                    print(f"\nConfiguration {idx+1}/{len(fdet_configs)}")
                    print("\n Training Parameters:")
                    for key, value in fdet_config.hparams.items():
                        print(f"  {key}: {value}")
                    
                    # run the training pipeline for the current config
                    with console_logger_train.capture_output():
                        print("\nStarting fault detection model training...")

                        train_pipeline = AnomalyDetectorTrainPipeline(data_config, fdet_config)
                        train_pipeline.train()

                        base_name = os.path.basename(train_pipeline.train_log_path) if train_pipeline.train_log_path else fdet_config.anom_config['anom_type']
                        print('\n' + 75*'=')
                        print(f"\nFault detection model '{base_name}' training completed.")

                    if fdet_config.is_log:
                        # save the captured output to a file
                        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
                        console_logger_train.save_to_file(file_path, script_name="fault_detection.train.py", base_name=base_name)

                    print('\n' + 75*'%')
    
    print('\n' + 75*'=')
    print('\n' + 75*'=')

    fdet_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_file_path = os.path.join(fdet_dir, "docs", f"sweep_{sweep_num}_console_output.txt")
    console_logger_sweep.save_to_file(sweep_file_path, script_name="fault_detection.multi_train.py", base_name=f"sweep_{sweep_num}")

    print(f"\nFault detection model sweep completed. Sweep log saved at: {sweep_file_path}")

if __name__ == "__main__":
    mtc = MultiTrainerConfig()
    
    multi_trainer_main(
        mtc.data_config_groups, 
        parallel_mode=mtc.parallel_mode, 
        max_workers=mtc.max_workers
        )
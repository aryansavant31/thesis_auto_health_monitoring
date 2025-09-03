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
from .settings.manager import TopologyEstimationInferSweepManager
from .infer import NRIInferPipeline, DecoderInferPipeline

def make_tp_results_dict(preds, data_config, model_name, results_dict):
    model_id = model_name.split('-')[0]

    node_group = model_id.split('(')[0]
    signal_group = model_id.split('(')[1].split('+')[0]
    set_id = model_id.split('+')[1].strip(')')

    all_nodes = data_config.signal_types['all_nodes']
    all_signal_groups = data_config.signal_types['all_signal_groups']

    results_dict[set_id][node_group][signal_group]['preds'] = preds
    results_dict[set_id][node_group][signal_group]['all_nodes'] = all_nodes
    results_dict[set_id][node_group][signal_group]['all_signal_groups'] = all_signal_groups

    return results_dict

def infer_single_config(args):
    """
    Infer a single configuration.

    Parameters
    ----------
    args : tuple
        (idx, tp_config, total_configs, framework, run_type, device)
    """
    idx, decoder_config, nri_config, total_configs, framework, run_type, device = args

    console_logger_infer = ConsoleLogger()

    print(f"\nConfiguration {idx+1}/{total_configs}")

    use_nri = False

    with console_logger_infer.capture_output():
        print(f"\nStarting {framework} model to {run_type}...")
        preds = {}

        if framework == 'decoder' or framework == 'full_tp':
            decoder_infer_pipeline = DecoderInferPipeline(decoder_config.data_config, decoder_config)

            preds = decoder_infer_pipeline.infer(device=device)
            base_name = f"{decoder_config.selected_model_num}/{os.path.basename(decoder_infer_pipeline.infer_log_path)}" if decoder_infer_pipeline.infer_log_path else f"{decoder_config.selected_model_num}/{run_type}"
            
            print('\n' + 75*'=')
            print(f"\n{run_type.capitalize()} of decoder model '{base_name}' completed.")

            if preds['dec/residual'] > decoder_config.residual_thresh:
                print(f"\nDecoder residual {preds['dec/residual']:,.4f} > {decoder_config.residual_thresh}. Hence using NRI model for topology prediction.")
                use_nri = True
            else:
                print(f"\nDecoder residual {preds['dec/residual']:,.4f} <= {decoder_config.residual_thresh}. Hence given topology to decoder is correct.")
                use_nri = False

            if decoder_config.is_log:
                # save the captured output to a file
                file_path = os.path.join(decoder_infer_pipeline.infer_log_path, "console_output.txt")
                console_logger_infer.save_to_file(file_path, script_name="topology_estimation.infer.py", base_name=base_name)


        if framework == 'nri' or (framework == 'full_tp' and use_nri):
            nri_infer_pipeline = NRIInferPipeline(nri_config.data_config, nri_config)

            preds_enc = nri_infer_pipeline.infer(device=device)
            preds.update(preds_enc)

            base_name = f"{nri_config.selected_model_num}/{os.path.basename(nri_infer_pipeline.infer_log_path)}" if nri_infer_pipeline.infer_log_path else f"{nri_config.selected_model_num}/{run_type}"
            
            print('\n' + 75*'=')
            print(f"\n{run_type.capitalize()} of nri model '{base_name}' completed.")

            if nri_config.is_log:
                # save the captured output to a file
                file_path = os.path.join(nri_infer_pipeline.infer_log_path, "console_output.txt")
                console_logger_infer.save_to_file(file_path, script_name="topology_estimation.infer.py", base_name=base_name)

    print('\n' + 75*'%')

    return {
        'idx': idx,
        'base_name': base_name,
        'model_name': base_name.split('/')[0],
        'data_config': decoder_config.data_config if framework != 'nri' else nri_config.data_config,
        'preds': preds if run_type == 'predict' else None
    }

def multi_inferer_main(framework, run_type, parallel_execution, max_workers=None):
    """
    Perform multi-configuration inference for NRI or decoder models.

    Parameters
    ----------
    framework : str
        The model framework to use ('nri' or 'decoder').
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
        print(f"\nStarting {framework} model inference sweep...")

        # get data configuration
        data_sweep = DataSweep(run_type=run_type)
        data_configs = data_sweep.get_sweep_configs()

        print(f"\nProcessing {len(data_configs)} data groups")
        data_sweep.print_sweep_summary()

        # get all inference configurations
        if framework == 'decoder':
            sweep_manager = TopologyEstimationInferSweepManager(
                data_configs=data_configs, 
                framework='decoder', 
                run_type=run_type
            )
            decoder_configs = sweep_manager.get_sweep_configs()
            sweep_manager.print_sweep_summary()
            infer_sweep_num = sweep_manager.infer_sweep_num

            n_configs = len(decoder_configs)
            nri_configs = range(n_configs)  # dummy range

        elif framework == 'nri':
            sweep_manager = TopologyEstimationInferSweepManager(
                data_configs=data_configs, 
                framework='nri', 
                run_type=run_type
            )
            nri_configs = sweep_manager.get_sweep_configs()
            sweep_manager.print_sweep_summary()
            infer_sweep_num = sweep_manager.infer_sweep_num

            n_configs = len(nri_configs)
            decoder_configs = range(n_configs)

        elif framework == 'full_tp':
            sweep_manager_dec = TopologyEstimationInferSweepManager(
                data_configs=data_configs, 
                framework='decoder',
                run_type=run_type
            )
            decoder_configs = sweep_manager_dec.get_sweep_configs()
            sweep_manager_dec.print_sweep_summary()

            sweep_manager_nri = TopologyEstimationInferSweepManager(
                data_configs=data_configs, 
                framework='nri',
                run_type=run_type
            )
            nri_configs = sweep_manager_nri.get_sweep_configs()
            sweep_manager_nri.print_sweep_summary()

            infer_sweep_num = sweep_manager_dec.infer_sweep_num
            
            assert infer_sweep_num == sweep_manager_nri.infer_sweep_num, "Number of decoder and NRI configs must be the same for full topology estimation."
            assert len(decoder_configs) == len(nri_configs), "Number of decoder and NRI configs must be the same for full topology estimation."

        else:
            raise ValueError(f"Invalid framework: {framework}. Choose 'nri', 'decoder' or 'full_tp'.")
        
        if parallel_execution:
            print(f"\nParallelizing {n_configs} {framework} configurations across {max_workers} workers...")
            
            device = 'cpu'
            print(f"\nSince running configs sequenctially, using device: {device}")

            config_args = [(idx, decoder_configs[idx], nri_configs[idx], n_configs, framework, run_type, device)
                            for idx in range(n_configs)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all configuration training tasks
                future_to_config = {
                    executor.submit(infer_single_config, arg): arg 
                    for arg in config_args
                }

                # Collect results as they complete
                config_results = []
                results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

                for future in concurrent.futures.as_completed(future_to_config):
                    try:
                        result = future.result()
                        config_results.append(result)
                        if result['preds'] is not None:
                            results_dict = make_tp_results_dict(result['preds'], result['data_config'], result['model_name'], results_dict)
                        print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")

                    except Exception as e:
                        config_arg = future_to_config[future]
                        print(f"  Config {config_arg[0] + 1} failed with error: {e}")

                # Sort results by index to maintain order
                config_results.sort(key=lambda x: x['idx'])

        else:
            print(f"\nRunning {n_configs} {framework} configurations sequentially (no parallelization)...")
            
            device = 'auto'
            print(f"\nSince running configs sequenctially, using device: {device}")

            config_args = [(idx, decoder_configs[idx], nri_configs[idx], n_configs, framework, run_type, device)
                           for idx in range(n_configs)]
            
            results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
            for arg in config_args:
                result = infer_single_config(arg)
                if result['preds'] is not None:
                    results_dict = make_tp_results_dict(result['preds'], result['data_config'], result['model_name'], results_dict)
                print(f"  Completed config {result['idx'] + 1}: {result['base_name']}")
            
        print('\n' + 75*'=')
        print('\n' + 75*'=')

        tp_dir = os.path.dirname(os.path.abspath(__file__))
        sweep_file_path = os.path.join(tp_dir, "docs", "sweep_logs", framework, f"{run_type}_sweeps")
        if not os.path.exists(sweep_file_path):
            os.makedirs(sweep_file_path)

        if run_type == 'predict':
            npy_file_path = os.path.join(sweep_file_path, f"tp_results_dict.npy")
            np.save(npy_file_path, results_dict)
            print(f"\n{framework.capitalize()} results dictionary saved at: {npy_file_path}")
        
        text_file_path = os.path.join(sweep_file_path, f"iswp_{infer_sweep_num}_output.txt")
        print(f"\n{framework.capitalize()} model '{run_type}' sweep completed. Sweep log saved at: {text_file_path}")

    console_logger_sweep.save_to_file(text_file_path, script_name="topology_estimation.multi_infer.py", base_name=f"iswp_{infer_sweep_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer multiple topology estimation models based on sweep configurations.")

    parser.add_argument('--framework', type=str,
                        choices=['nri', 'decoder', 'full_tp'],
                        default='full_tp',
                        required=True, help="Model framework: nri, decoder or full_tp")
    
    parser.add_argument('--run-type', type=str, 
                        choices=['custom_test', 'predict'],
                        default='predict',
                        required=True, help="Run type: custom_test or predict")
    
    parser.add_argument('--parallel', action='store_true',
                    help="Enable parallel execution of inference configurations.")
    
    parser.add_argument('--max-workers', type=int, default=8,
                    help="Maximum number of workers for parallel execution. Default is 8.")
    
    args = parser.parse_args()

    multi_inferer_main(framework=args.framework,
                        run_type=args.run_type, 
                        parallel_execution=args.parallel, 
                        max_workers=args.max_workers)
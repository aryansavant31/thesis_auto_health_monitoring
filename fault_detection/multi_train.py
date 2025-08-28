import os, sys

# global imports
from data.config import DataConfig
from data.datasets.bearing.groups import BERGroupMaker
from console_logger import ConsoleLogger

# local imports
from .settings.manager import AnomalyDetectorSweepManager
from .single_train import AnomalyDetectorTrainPipeline

def multi_trainer_main(sweep_num):
    """
    Main function to run multiple training sessions for the anomaly detector.
    """
    # Parameters
    console_logger_sweep = ConsoleLogger()
    console_logger_train = ConsoleLogger()

    data_config = DataConfig(run_type="train")

    all_groups = [BERGroupMaker().gb_acc]

    for group in all_groups:
        data_config.signal_types = group

        sweep_config = AnomalyDetectorSweepManager(data_config, make_model_num=True)

        with console_logger_sweep.capture_output():
            print("\nStarting fault detection model sweep...")
            sweep_config.print_sweep_summary()

            # get all training configurations
            fdet_configs = sweep_config.get_sweep_configs()


            # train models for each configuration

            for idx, fdet_config in enumerate(fdet_configs):
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
    sweep_num = 1
    multi_trainer_main(sweep_num)

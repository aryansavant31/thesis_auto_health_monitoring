import os, sys

# other imports
from torch.utils.tensorboard import SummaryWriter
import argparse

# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger

# local imports
from .settings.manager import FaultDetectorInferManager, get_model_pickle_path
from .detector import FaultDetector, TrainerFaultDetector


class FaultDetectorInferPipeline:
    def __init__(self, data_config:DataConfig, fdet_config:FaultDetectorInferManager):
        """
        Initialize the anomaly detector prediction main class.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        fdet_config : FaultDetectorPredictManager
            The anomaly detector prediction configuration object.
        """
        self.data_config = data_config
        self.fdet_config = fdet_config
        self.data_preprocessor = DataPreprocessor("fault_detection")

    def infer(self):
        """
        Main method to infer the anomaly detector.

        Returns
        -------
        preds : dict, optional
            A dictionary containing:
            - `pred_labels`: torch.Tensor of shape (n_samples,) containing predicted classes (0 for normal, 1 for anomaly)
            - `scores`: torch.Tensor of shape (n_samples,) containing anomaly scores for each sample
            - `reps`: torch.Tensor of shape (n_samples,) containing rep numbers for each sample
            
            (Only returned if run_type is 'predict').
        """
        # 1. Load data
        custom_loader = self.data_preprocessor.get_custom_data_package(
            self.data_config, self.fdet_config.batch_size)

        # 2. Load the anomaly detector model
        fault_detector = self._load_model()

        # 3. Infer using the anomaly detector model
        logger = self._prep_for_inference()

        tester = TrainerFaultDetector(logger=logger)
        if self.fdet_config.run_type == 'custom_test':
            tester.test(fault_detector, custom_loader)

            # plot the test results
            for plot_name, plot_config in self.fdet_config.test_plots.items():
                if plot_config[0]:
                    getattr(tester, plot_name.split('-')[0])(**plot_config[1])
                
        elif self.fdet_config.run_type == 'predict':
            preds = tester.predict(fault_detector, custom_loader)
            
            # plot the predict results
            for plot_name, plot_config in self.fdet_config.test_plots.items():
                if plot_config[0]:
                    getattr(tester, plot_name.split('-')[0])(**plot_config[1])

            return preds

    def _load_model(self):
        """
        Load the anomaly detector model from the checkpoint path.

        Parameters
        ----------
        custom_data_stats : dict
            Statistics of the custom data.

        Returns
        -------
        FaultDetector
            The loaded anomaly detector model.
        """
        fault_detector = FaultDetector.load_from_pickle(self.fdet_config.selected_model_path)

        # update the model with custom data statistics and domain config
        fault_detector.domain_config = self.fdet_config.domain_config
        fault_detector.nok_percentage = self.fdet_config.nok_percentage

        # update hparams with new values
        fault_detector.hparams.update(self.fdet_config.infer_hparams)
        fault_detector.hparams.update({
            f"max_timesteps/{self.fdet_config.run_type}" : f"{int(self.data_config.max_timesteps):,}"
            })

        print(f"\nAnomaly detector model loaded for '{self.fdet_config.run_type}' from {self.fdet_config.selected_model_path}")
        print(f"\nModel type: {type(fault_detector.anom_model).__name__}, Model ID: {fault_detector.hparams['model_id']}, No. of input features req.: {fault_detector.anom_model.n_features_in_ if hasattr(fault_detector.anom_model, 'n_features_in_') else 'Unknown'}")
        print('\n' + 75*'-')
        
        return fault_detector    
    
    def _prep_for_inference(self):
        """
        Prepare the logger for inference.
        """
        if self.fdet_config.is_log:
            self.infer_log_path = self.fdet_config.get_infer_log_path()

            self.fdet_config.save_infer_params()
            logger = SummaryWriter(log_dir=self.infer_log_path)

            # log all the attributes of train_config
            # formatted_params = "\n".join([f"{key}: {value}" for key, value in self.fdet_config.__dict__.items()])
            # logger.add_text(os.path.basename(self.infer_log_path), formatted_params)

            # log dataset selected
            data_text = self.data_preprocessor.get_data_selection_text()
            base_name = f"{self.fdet_config.selected_model_num} + {os.path.basename(self.infer_log_path)}"
            logger.add_text(base_name, data_text)

            print(f"\n{self.fdet_config.run_type.capitalize()} environment set. {self.fdet_config.run_type.capitalize()} will be logged at: {self.infer_log_path}")

        else:
            self.infer_log_path = None
            logger = None
            print(f"\n{self.fdet_config.run_type.capitalize()} environment set. Logging is disabled.")

        print('\n' + 75*'-')
        return logger
    
if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()
    parser = argparse.ArgumentParser(description="Infer the anomaly detector model.")

    parser.add_argument('--run-type', type=str, 
                    choices=['custom_test', 'predict'],
                    default='predict',
                    required=True, help="Run type: custom_test or predict")
    
    args = parser.parse_args()

    data_config = DataConfig(run_type=args.run_type)
    fdet_config = FaultDetectorInferManager(data_config, run_type=args.run_type)

    with console_logger.capture_output():
        print(f"\nStarting anomaly detector to {args.run_type}...")

        infer_pipeline = FaultDetectorInferPipeline(data_config, fdet_config)
        if args.run_type == 'custom_test':
            infer_pipeline.infer()

        elif args.run_type == 'predict':
            preds = infer_pipeline.infer()
        
        base_name = f"{fdet_config.selected_model_num}/{os.path.basename(infer_pipeline.infer_log_path)}" if infer_pipeline.infer_log_path else f"{fdet_config.selected_model_num}/{args.run_type}"
        print('\n' + 75*'=')
        print(f"\n{args.run_type.capitalize()} of anomaly detector '{base_name}' completed.")

    if fdet_config.is_log:
        # save the captured output to a file
        file_path = os.path.join(infer_pipeline.infer_log_path, "console_output.txt")
        console_logger.save_to_file(file_path, script_name="fault_detection.infer.py", base_name=base_name)
        
import sys, os

# other imports
import inspect
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

# global imports
from data.config import DataConfig
from data.prep import DataPreprocessor
from console_logger import ConsoleLogger

# local imports
from .settings.manager import NRITrainManager, DecoderTrainManager, get_model_ckpt_path
from .relations import RelationMatrixMaker
from .utils.custom_loader import CombinedDataLoader
from .nri import NRI
from .decoder import Decoder
from .encoder import Encoder

class TopologyEstimationTrainHelper:
    """
    Base class for NRI and Decoder training pipelines.
    """
    def __init__(self, data_config:DataConfig, tp_config:NRITrainManager|DecoderTrainManager):
        self.data_config = data_config
        self.tp_config = tp_config

        self.data_preprocessor = DataPreprocessor("topology_estimation")
        self.rm = RelationMatrixMaker(self.tp_config.spf_config)

    def load_training_data(self):
        """
        Load the training, validation, and test data.

        Attributes
        ----------
        train_loader : DataLoader
            DataLoader for the training data.
        test_loader : DataLoader
            DataLoader for the test data.
        val_loader : DataLoader
            DataLoader for the validation data.
        train_data_stats : dict
            Statistics of the training data.
        test_data_stats : dict
            Statistics of the test data.
        val_data_stats : dict
            Statistics of the validation data.
        """
        train_data, test_data, val_data = self.data_preprocessor.get_training_data_package(
            self.data_config, 
            batch_size=self.tp_config.batch_size,
            train_rt=self.tp_config.train_rt,
            test_rt=self.tp_config.test_rt,
            val_rt=self.tp_config.val_rt,
            num_workers=self.tp_config.num_workers,
            )
        # unpack data_loaders and data_stats
        self.train_loader, self.train_data_stats = train_data
        self.test_loader, self.test_data_stats = test_data
        self.val_loader, self.val_data_stats = val_data

    # def load_relation_matrix_loaders(self):
    #     """
    #     Load the relation matrix loaders for training, validation, and test data.

    #     Attributes
    #     ----------
    #     rel_loader_train : DataLoader
    #         DataLoader for the relation matrices of the training data.
    #     rel_loader_test : DataLoader
    #         DataLoader for the relation matrices of the test data.
    #     rel_loader_val : DataLoader
    #         DataLoader for the relation matrices of the validation data.
    #     """
        

    def get_encoder_params(self):
        """
        Get the encoder parameters required for initializing the NRI model.

        Returns
        -------
        enc_model_params : dict
            Dictionary containing the encoder model parameters.
        """
        # encoder params
        req_enc_model_params = [param for param in Encoder().__dict__.keys()]

        enc_model_params = {
            key.removeprefix('enc_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('enc_') in req_enc_model_params and key not in ['hyperparams']
        }

        # get n_comps and n_dims for encoder and decoder
        pre_enc = Encoder()
        for key, value in enc_model_params.items():
            setattr(pre_enc, key, value)
        pre_enc.set_run_params(data_config=self.data_config, data_stats=self.train_data_stats)
        pre_enc.init_input_processors(is_verbose=False)
        n_comps, n_dims = pre_enc.process_input_data(next(iter(self.train_loader))[0], get_data_shape=True)

        enc_model_params['n_comps'] = n_comps
        enc_model_params['n_dims'] = n_dims

        print("\n" + 6*"<" + " ENCODER PARAMETERS " + 6*">")
        print("Encoder model parameters:")
        print(25 * "-")
        for key, value in enc_model_params.items():
            print(f"{key}: {value}")

        # print('\n' + 35*'-')

        return enc_model_params
        
    def get_decoder_params(self):
        """
        Get the decoder parameters
        """
        # decoder params
        req_dec_model_params = [param for param in Decoder().__dict__.keys()] 
        req_dec_run_params = inspect.signature(Decoder.set_run_params).parameters.keys()

        dec_model_params = {
            key.removeprefix('dec_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('dec_') in req_dec_model_params and key not in ['hyperparams']
        }
        dec_run_params = {
            key.removeprefix('dec_'): value for key, value in self.tp_config.__dict__.items() if key.removeprefix('dec_') in req_dec_run_params and key not in ['data_config']
        }
        dec_model_params['n_dims'] = next(iter(self.train_loader))[0].shape[3]

        print("\n" + 6*"<" + " DECODER PARAMETERS " + 6*">")
        print("\nDecoder model parameters:")
        print(25 * "-")
        for key, value in dec_model_params.items():
            print(f"{key}: {value}")

        print("\nDecoder run parameters:")
        print(25 * "-")
        for key, value in dec_run_params.items():
            print(f"{key}: {value}")

        # print('\n' + 35*'-')

        return dec_model_params, dec_run_params
    
    def _prep_for_training(self, n_dims, n_comps=None):
        """
        Prepare the training environment before starting the training process.
        """
        # if logging enabled, save parameters and initialize TensorBoard logger
        if self.tp_config.is_log:
            self.train_log_path = self.tp_config.get_train_log_path(n_dims, n_comps=n_comps)

            # if continue training, load ckpt path of untrained model 
            if self.tp_config.continue_training:
                ckpt_path = get_model_ckpt_path(self.train_log_path)
            else:
                ckpt_path = None

            self.tp_config.save_params()

            # define ckeckpoint callback to save the best model
            if self.tp_config.framework == 'nri':
                monitor_metric = 'nri/val_loss'
            elif self.tp_config.framework == 'decoder':
                monitor_metric = 'val_loss'

            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.train_log_path, 'checkpoints'),
                filename='best-model-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                monitor=monitor_metric,
                mode='min'
            )

            # define logger
            train_logger = TensorBoardLogger(os.path.dirname(self.train_log_path), name="", version=os.path.basename(self.train_log_path))

            # log all the attributes of train_config
            # formatted_params = "\n".join([f"{key}: {value}" for key, value in self.tp_config.__dict__.items()])
            # train_logger.experiment.add_text(os.path.basename(self.train_log_path), formatted_params)

            # log dataset selected
            data_text = self.data_preprocessor.get_data_selection_text()
            rel_text = self.rm.get_relation_matrices_summary()
            data_text += rel_text

            train_logger.experiment.add_text(f"{os.path.basename(self.train_log_path)} + train", data_text)

            print(f"\nTraining environment set. Training will be logged at: {self.train_log_path}")

        else:
            self.train_log_path = None
            train_logger = None
            checkpoint_callback = None
            ckpt_path = None
            print("\nTraining environment set. Logging is disabled.")

        print('\n' + 75*'-')

        return train_logger, checkpoint_callback, ckpt_path
    
    def _prep_for_testing(self):
        """
        Prepare the testing environment before starting the testing process.
        """
        # if logging enabled, initialize TensorBoard logger
        if self.tp_config.is_log:
            # get log path to save trained model
            test_log_path = self.tp_config.get_test_log_path()
            test_logger = TensorBoardLogger(os.path.dirname(test_log_path), name="", version=os.path.basename(test_log_path))
            print(f"\nTesting environment set. Testing will be logged at: {test_log_path}")
        else:
            test_logger = None
            print("\nTesting environment set. Logging is disabled.")
        
        return test_logger

class NRITrainPipeline(TopologyEstimationTrainHelper):
    def __init__(self, data_config:DataConfig, nri_config:NRITrainManager):
        """
        Initialize the NRI training main class.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        nri_config : NRITrainManager
            The NRI training configuration object.
        """
        super().__init__(data_config, nri_config)

    def train(self, device='auto', fast_dev_run=False):
        """
        Main method to train the NRI model.
        """
    # 1. Load data
        self.load_training_data()

        # load relation matrices
        rec_rel, send_rel = self.rm.get_relation_matrix(self.train_loader)

    # 2. Initialize the NRI model
        enc_model_params = self.get_encoder_params()
        dec_model_params, dec_run_params = self.get_decoder_params()

        nri_model = self._init_nri_model(
            enc_model_params,
            dec_model_params, dec_run_params,
            rec_rel, send_rel, self.train_data_stats
            )

    # 3. Train the NRI model
        train_logger, checkpoint_callback, ckpt_path = self._prep_for_training(enc_model_params['n_comps'], dec_model_params['n_dims'])
        
        trainer = Trainer(
            accelerator=device,
            callbacks=[checkpoint_callback],
            logger=train_logger,
            max_epochs=1 if fast_dev_run else self.tp_config.max_epochs,
            enable_progress_bar=False,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            limit_train_batches=1 if fast_dev_run else None,
            limit_val_batches=1 if fast_dev_run else None
            )

        trainer.fit(model=nri_model, train_dataloaders=self.train_loader, 
                    val_dataloaders=self.val_loader, ckpt_path=ckpt_path)
        
        print('\n' + 75*'-')

        if fast_dev_run:
            print("\nFast dev run completed. Exiting without testing.")
            return

    # 4. Test the trained NRI model
        print("\nTESTING TRAINED NRI MODEL...")
        trained_nri_model = self._load_model(dec_run_params, rec_rel, send_rel, self.test_data_stats)
        
        test_logger = self._prep_for_testing()

        tester = Trainer(
            accelerator=device,
            logger=test_logger)
        
        tester.test(model=trained_nri_model, dataloaders=self.test_loader)


    def _init_nri_model(
        self, enc_model_params, 
        dec_model_params, dec_run_params, 
        rec_rel, send_rel, train_data_stats
    ):
        """
        Initialize the NRI model with the given parameters.
        """
        # prep hyperparams
        self.tp_config.hyperparams.update({
            'n_comps': str(int(enc_model_params['n_comps'])),
            'n_dims': str(int(dec_model_params['n_dims'])),
            'n_nodes': str(int(next(iter(self.train_loader))[0].shape[1])),
            'max_timesteps': f"{int(self.data_config.max_timesteps):,}"  
            })
            
        nri_model = NRI()
        nri_model.encoder_model_params = enc_model_params
        nri_model.decoder_model_params = dec_model_params
        nri_model.build_model()

        # set other params
        nri_model.set_hyperparams(self.tp_config.hyperparams)
        nri_model.set_input_graph(rec_rel, send_rel)
        
        nri_model.set_run_params(
            dec_run_params=dec_run_params, 
            data_config=self.data_config, 
            data_stats=train_data_stats, 
            init_temp=self.tp_config.init_temp,
            min_temp=self.tp_config.min_temp,
            decay_temp=self.tp_config.decay_temp,
            is_hard=self.tp_config.is_hard
            )
        
        nri_model.set_training_params(
            lr_enc=self.tp_config.lr_enc,
            lr_dec=self.tp_config.lr_dec, 
            is_beta_annealing=self.tp_config.is_beta_annealing,
            final_beta=self.tp_config.final_beta,
            warmup_frac_beta=self.tp_config.warmup_frac_beta,
            optimizer=self.tp_config.optimizer,
            add_const_kld=self.tp_config.add_const_kld,
            loss_type_enc=self.tp_config.loss_type_enc,
            loss_type_dec=self.tp_config.loss_type_dec,
            prior = self.tp_config.prior
            )

        # print model info
        print("\n" + 75*'-')
        print("\nNRI Model Initialized with the following configurations:")
        nri_model.print_model_info()
        print('\n' + 75*'-')

        return nri_model
    
    def _load_model(self, dec_run_params, rec_rel, send_rel, test_data_stats):
        """
        Load the trained NRI model from the checkpoint path.
        """
        trained_nri_model = NRI.load_from_checkpoint(get_model_ckpt_path(self.train_log_path))

        trained_nri_model.set_input_graph(rec_rel, send_rel)

        trained_nri_model.set_run_params(
            dec_run_params=dec_run_params, 
            data_config=self.data_config, 
            data_stats=test_data_stats, 
            init_temp=self.tp_config.init_temp,
            min_temp=self.tp_config.min_temp,
            decay_temp=0, 
            is_hard=self.tp_config.is_hard
            )

        print("\nTrained NRI Model Loaded for testing.")
        return trained_nri_model
    
    
class DecoderTrainPipeline(TopologyEstimationTrainHelper):
    def __init__(self, data_config:DataConfig, decoder_config:DecoderTrainManager):
        """
        Initialize the Decoder training main class.

        Parameters
        ----------
        data_config : DataConfig
            The data configuration object.
        decoder_config : DecoderTrainManager
            The Decoder training configuration object.
        """
        super().__init__(data_config, decoder_config)

    def train(self, device='auto', fast_dev_run=False):
        """
        Main method to train the Decoder model.
        """
    # 1. Load data
        self.load_training_data()

        # load relation matrices
        rec_rel, send_rel = self.rm.get_relation_matrix(self.train_loader)

    # 2. Initialize the Decoder model
        dec_model_params, dec_run_params = self.get_decoder_params()
        
        decoder_model = self._init_decoder_model(
            dec_model_params, dec_run_params, 
            rec_rel, send_rel, self.train_data_stats)

    # 3. Train the Decoder model
        train_logger, checkpoint_callback, ckpt_path = self._prep_for_training(dec_model_params['n_dims'])
        
        trainer = Trainer(
            accelerator=device,
            callbacks=[checkpoint_callback], 
            logger=train_logger,
            max_epochs=1 if fast_dev_run else self.tp_config.max_epochs, 
            enable_progress_bar=True,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            limit_train_batches=1 if fast_dev_run else None,
            limit_val_batches=1 if fast_dev_run else None 
            )
        trainer.fit(model=decoder_model, train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader, ckpt_path=ckpt_path)
        
        print('\n' + 75*'-')

        if fast_dev_run:
            print("\nFast dev run completed. Exiting without testing.")
            return

    # 4. Test the trained Decoder model
        print("\nTESTING TRAINED DECODER MODEL...")
        trained_decoder_model = self._load_model(dec_run_params, rec_rel, send_rel, self.test_data_stats)

        test_logger = self._prep_for_testing()
        tester = Trainer(
            accelerator=device,
            logger=test_logger)
        
        tester.test(model=trained_decoder_model, dataloaders=self.test_loader)


    def _init_decoder_model(
        self, dec_model_params:dict, dec_run_params:dict, 
        rec_rel, send_rel, train_data_stats
    ):
        """
        Initialize the Decoder model with the given parameters.
        """
        # prep hyperparams
        self.tp_config.hyperparams.update({
            'n_dims': str(int(dec_model_params['n_dims'])),
            'n_nodes': str(int(next(iter(self.train_loader))[0].shape[1])),
            'max_timesteps': f"{int(self.data_config.max_timesteps):,}"  
        })

        # initialize decoder model
        decoder_model = Decoder()

        for key, value in dec_model_params.items():
            setattr(decoder_model, key, value)
        decoder_model.build_model()

        # set other params
        decoder_model.set_hyperparams(self.tp_config.hyperparams)
        decoder_model.set_input_graph(rec_rel, send_rel, make_edge_matrix=True, batch_size=self.train_loader.batch_size)

        decoder_model.set_run_params(
            **dec_run_params, 
            data_config=self.data_config, 
            data_stats=train_data_stats
            )
        
        # set training params
        decoder_model.set_training_params(
            lr=self.tp_config.lr, 
            optimizer=self.tp_config.optimizer,
            loss_type=self.tp_config.loss_type,
            momentum=self.tp_config.momentum
            )

        print("\n" + 75*'-')
        print("\nDecoder Model Initialized with the following configurations:")
        print("\nDecoder Model Summary:")
        print(decoder_model)

        print('\n' + 75*'-')
        return decoder_model
    
    def _load_model(self, dec_run_params, rec_rel, send_rel, test_data_stats):
        """
        Load the trained Decoder model from the checkpoint path.
        """
        trained_decoder_model = Decoder.load_from_checkpoint(get_model_ckpt_path(self.train_log_path))

        trained_decoder_model.set_input_graph(rec_rel, send_rel, make_edge_matrix=True, batch_size=self.train_loader.batch_size)

        trained_decoder_model.set_run_params(
            **dec_run_params, 
            data_config=self.data_config, 
            data_stats=test_data_stats
            )

        print("\nTrained Decoder Model Loaded for testing.")
        return trained_decoder_model
        

if __name__ == "__main__":
    # create console logger to log all the outputs in terminal
    console_logger = ConsoleLogger()
    parser = argparse.ArgumentParser(description="Train NRI and decoder models.")

    parser.add_argument('--framework', type=str, 
                    choices=['nri', 'decoder'],
                    default='nri',
                    required=True, help="Framework to train: nri or decoder")
    
    parser.add_argument('--fast-dev-run', action='store_true',
                    help="If set, runs a single batch through training and validation to quickly check for any errors.")
    
    args = parser.parse_args()
    
    data_config = DataConfig(run_type='train')
    if args.framework == 'nri':
        nri_config = NRITrainManager(data_config)
    elif args.framework == 'decoder':
        decoder_config = DecoderTrainManager(data_config)

    
    with console_logger.capture_output():
        print(f"\nStarting {args.framework} model training...")

        if args.framework == 'nri':
            train_pipeline = NRITrainPipeline(data_config, nri_config)
        elif args.framework == 'decoder':
            train_pipeline = DecoderTrainPipeline(data_config, decoder_config)
        else:
            raise ValueError(f"Invalid framework: {args.framework}. Choose 'nri' or 'decoder'.")
        
        train_pipeline.train(fast_dev_run=args.fast_dev_run)
        
        base_name = os.path.basename(train_pipeline.train_log_path) if train_pipeline.train_log_path else f"{args.framework}_model"
        print('\n' + 75*'=')
        print(f"\n{args.framework.capitalize()} model '{base_name}' training completed.")

    if train_pipeline.train_log_path:
        # save the captured output to a file
        file_path = os.path.join(train_pipeline.train_log_path, "console_output.txt")
        console_logger.save_to_file(file_path, script_name="topology_estimation.train.py", base_name=base_name)

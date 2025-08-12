import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR) if ROOT_DIR not in sys.path else None

import torch


from data.prep import DataPreprocessor
from data.config import DataConfig
from pytorch_lightning.utilities import rank_zero_only

def load_stuff():
    data_config = DataConfig(run_type='train')
    data_preprocessor = DataPreprocessor(package='topology_estimation')

    # load dataloaders (runs in all ranks)
    train_package, test_package, val_package = data_preprocessor.get_training_data_package(
        data_config, train_rt=0.8, test_rt=0.1, val_rt=0.1, num_workers=0
    )

    train_loader, train_data_stats = train_package
    test_loader, test_data_stats = test_package
    val_loader, val_data_stats = val_package

    dataiter = iter(train_loader)
    data = next(dataiter)

    n_nodes = data[0].shape[1]
    n_timesteps = data[0].shape[2]
    n_dims = data[0].shape[3]

    from settings.manager import NRITrainManager
    nri_config = NRITrainManager(data_config)

    from graph_structures import RelationMatrixMaker
    rm = RelationMatrixMaker(nri_config.spf_config, n_nodes)
    rel_loader = rm.get_relation_matrix_loader(train_loader)

    from encoder import Encoder
    from decoder import Decoder
    import inspect

    # Encoder setup
    req_enc_model_params = inspect.signature(Encoder.__init__).parameters.keys()
    req_enc_run_params = inspect.signature(Encoder.set_run_params).parameters.keys()
    enc_model_params = {key.removesuffix('_enc'): value for key, value in nri_config.__dict__.items() if key.removesuffix('_enc') in req_enc_model_params}
    enc_run_params = {key.removeprefix('enc_'): value for key, value in nri_config.__dict__.items() if key.removeprefix('enc_') in req_enc_run_params}
    enc_run_params['data_stats'] = train_data_stats

    # Decoder setup
    req_dec_model_params = inspect.signature(Decoder.__init__).parameters.keys()
    req_dec_run_params = inspect.signature(Decoder.set_run_params).parameters.keys()
    dec_model_params = {key.removesuffix('_dec'): value for key, value in nri_config.__dict__.items() if key.removesuffix('_dec') in req_dec_model_params}
    dec_run_params = {key.removeprefix('dec_'): value for key, value in nri_config.__dict__.items() if key.removeprefix('dec_') in req_dec_run_params}
    dec_run_params['data_stats'] = train_data_stats

    # Get n_comps and n_dims
    none_dict = {param: None for param in inspect.signature(Encoder).parameters.keys()}
    pre_enc = Encoder(**none_dict)
    pre_enc.set_run_params(**enc_run_params)
    n_comps, n_dims = pre_enc.process_input_data(data[0], get_data_shape=True)
    enc_model_params['n_comps'] = n_comps
    enc_model_params['n_dims'] = n_dims
    dec_model_params['n_dims'] = n_dims

    from nri import NRI
    nri_model = NRI(enc_model_params, dec_model_params)
    nri_model.set_run_params(enc_run_params, dec_run_params, nri_config.temp, nri_config.is_hard)
    nri_model.set_training_params()

    # Logging (only in rank 0)
    logger = None
    @rank_zero_only
    def create_logger():
        train_log_path = nri_config.get_train_log_path(
            n_comps=enc_model_params['n_comps'],
            n_dim=dec_model_params['n_dims']
        )
        if nri_config.is_log:
            nri_config.save_params()
            from pytorch_lightning.loggers import TensorBoardLogger
            return TensorBoardLogger(
                os.path.dirname(train_log_path),
                name="",
                version=os.path.basename(train_log_path)
            )
        return None

    logger = create_logger()

    return nri_model, train_loader, rel_loader, logger, nri_config

def main():
    nri_model, train_loader, rel_loader, logger, nri_config = load_stuff()

    from utils.custom_loader import CombinedDataLoader
    from pytorch_lightning import Trainer

    trainer = Trainer(
        max_epochs=nri_config.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        accelerator='cpu',
        devices=4,
        strategy='ddp'
    )

    trainer.fit(
        model=nri_model,
        train_dataloaders=CombinedDataLoader(train_loader, rel_loader)
    )

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # safer for DDP on CPU
    main()
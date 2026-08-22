import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from data.datamodule import HDF5DataModule
from models.exp_gtcrn import GTCRNExp # <-- CHANGE: Import GTCRNExp
from models.gtcrn_no_rnn import GTCRN       # <-- CHANGE: Import GTCRN
from typing import Optional
import yaml

EXP_NAME='GTCRN' # <-- CHANGE: Update experiment name

def setup_logging(tb_log_dir: str, version_id: Optional[int]= None):
    # This function can remain the same
    if version_id is None:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False)
        version_id = int((tb_logger.log_dir).split('_')[-1])
    else:
        tb_logger = pl_loggers.TensorBoardLogger(tb_log_dir, name=EXP_NAME, log_graph=False, version=version_id)
    return tb_logger, version_id

def load_model(ckpt_file: str,
               _config):
    # This function needs to use GTCRNExp
    init_params = GTCRNExp.get_init_params(_config) # Assuming get_init_params is a helper in the base class
    model = GTCRNExp.load_from_checkpoint(ckpt_file, **init_params)
    model.to('cuda')
    return model

def get_trainer(devices, logger, max_epochs, gradient_clip_val, gradient_clip_algorithm, strategy, accelerator):
    # First, define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',          # 'min' for loss, 'max' for metrics like SI-SNR
        save_top_k=3,        # Save only the single best model
        save_last=True,      # Also save a 'last.ckpt' for easy resuming
        dirpath=logger.log_dir, # Save to a 'checkpoints' subfolder
        filename='best-model-{epoch:02d}-{val/loss:.2f}', # Filename format
        auto_insert_metric_name=False
    )
    
    # This function can remain the same
    return pl.Trainer(enable_model_summary=True,
                         logger=logger,
                         devices=devices,
                         log_every_n_steps=1,
                         max_epochs=max_epochs,
                         gradient_clip_val=gradient_clip_val,
                         gradient_clip_algorithm=gradient_clip_algorithm,
                         strategy=strategy,
                         accelerator=accelerator,
                         callbacks=[checkpoint_callback, ModelSummary(max_depth=2)])

if __name__=="__main__":

    # <-- CHANGE: Point to the new config file
    with open('config/gtcrn_config.yaml') as config_file:
        config = yaml.safe_load(config_file)

    ## REPRODUCIBILITY
    pl.seed_everything(config.get('seed', 0), workers=True)

    ## LOGGING
    tb_logger, version = setup_logging(config['logging']['tb_log_dir'])

    ## DATA
    data_config = config['data']
    stft_length = data_config.get('stft_length_samples', 512)
    stft_shift = data_config.get('stft_shift_samples', 256)
    dm = HDF5DataModule(**data_config)

    ## CONFIGURE EXPERIMENT
    ckpt_file = config['training'].get('resume_ckpt', None)
    if not ckpt_file is None:
        exp = load_model(ckpt_file, config)
    else:
        # <-- CHANGE: Instantiate GTCRN and GTCRNExp
        model = GTCRN(**config['network']) 
        exp = GTCRNExp(model=model,
                       stft_length=stft_length,
                       stft_shift=stft_shift,
                       **config['experiment'])

    ## TRAIN
    trainer = get_trainer(logger=tb_logger, **config['training'])
    trainer.fit(exp, dm)
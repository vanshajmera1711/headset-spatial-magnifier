import torch
from typing import Optional
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint
from data.datamodule import HDF5DataModule
from models.exp_gtcrn import GTCRNExp
from models.gtcrn import GTCRN


torch.set_float32_matmul_precision("high")

EXP_NAME = "GTCRN"


def setup_logging(tb_log_dir: str, version_id: Optional[int] = None):
  if version_id is None:
    tb_logger = pl_loggers.TensorBoardLogger(
        tb_log_dir, name=EXP_NAME, log_graph=False
    )
    version_id = int((tb_logger.log_dir).split("_")[-1])
  else:
    tb_logger = pl_loggers.TensorBoardLogger(
        tb_log_dir, name=EXP_NAME, log_graph=False, version=version_id
    )
  return tb_logger, version_id


def load_model(ckpt_file: str, config: dict):
  """Loads model weights directly from the checkpoint file."""
  # If GTCRNExp requires a 'model' argument in __init__:
  model = GTCRN(**config["network"])
  exp = GTCRNExp.load_from_checkpoint(
      ckpt_file,
      model=model,
      stft_length=config["data"].get("stft_length_samples", 512),
      stft_shift=config["data"].get("stft_shift_samples", 256),
      **config["experiment"],
  )
  return exp


def get_trainer(
    devices,
    logger,
    max_epochs,
    gradient_clip_val,
    gradient_clip_algorithm,
    strategy,
    accelerator,
    precision="16-mixed",
    **kwargs
):
  checkpoint_callback = ModelCheckpoint(
      monitor="val/loss",
      mode="min",
      save_top_k=3,
      save_last=True,
      dirpath=logger.log_dir,
      filename="best-model-{epoch:02d}-{val/loss:.2f}",
      auto_insert_metric_name=False,
  )

  return pl.Trainer(
      enable_model_summary=True,
      logger=logger,
      devices=devices,
      log_every_n_steps=1,
      max_epochs=max_epochs,
      gradient_clip_val=gradient_clip_val,
      gradient_clip_algorithm=gradient_clip_algorithm,
      strategy=strategy,
      accelerator=accelerator,
      precision=precision,
      callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
  )


if __name__ == "__main__":
  with open("config/gtcrn_config.yaml") as config_file:
    config = yaml.safe_load(config_file)

  ## REPRODUCIBILITY
  pl.seed_everything(config.get("seed", 0), workers=True)

  ## LOGGING
  tb_logger, version = setup_logging(config["logging"]["tb_log_dir"])

  ## DATA
  data_config = config["data"]
  dm = HDF5DataModule(**data_config)

  ## CONFIGURE EXPERIMENT / LOAD CHECKPOINT
  ckpt_file = config['training'].get('resume_ckpt', None)
  if ckpt_file:
    print(f"Loading weights from checkpoint: {ckpt_file}")
    exp = load_model(ckpt_file, config)
  else:
    model = GTCRN(**config["network"])
    exp = GTCRNExp(
        model=model,
        stft_length=data_config.get("stft_length_samples", 512),
        stft_shift=data_config.get("stft_shift_samples", 256),
        **config["experiment"],
    )

  ## TRAIN
  trainer = get_trainer(logger=tb_logger, **config["training"])
  trainer.fit(exp, dm)
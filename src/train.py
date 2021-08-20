import os
import json
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
import torch
from torchvision import transforms

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init transforms
    log.info("Instantiating transforms")
    # if "train_transforms" in config:
    #     train_transforms = hydra.utils.instantiate(config.train_transforms)
    # else:
    #     train_transforms = None
    # if "test_transforms" in config:
    #     test_transforms = hydra.utils.instantiate(config.test_transforms)
    # else:
    #     test_transforms = None
    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    # Init Optimizer and Scheduler
    log.info(f"Instantiating optimizer <{config.optimizer._target_}> and scheduler <{config.scheduler._target_}>")

    if "optimizer_new_kws" in config:
        opt_params = [
            {
                "params": model.model.parameters(),
                "lr": config.optimizer_bb_kws.lr,
                "weight_decay": config.optimizer_bb_kws.weight_decay
            },
            {
                "params": model.enc.parameters()
            },
            {
                "params": model.pos_embed.parameters(),
                "weight_decay": 0.0
            },
            {
                "params": model.fc1.parameters()
            }
        ]
        optimizer = hydra.utils.instantiate(config.optimizer, params=opt_params)

    else:
        optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    scheduler = hydra.utils.instantiate(config.scheduler, optimizer=optimizer)
    scheduler_interval = config.scheduler_interval

    def configure_optimizers():
        return [optimizer], [{'scheduler': scheduler, 'interval': scheduler_interval}]
    model.configure_optimizers = configure_optimizers

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
        optimizer=optimizer,
        scheduler=scheduler
    )

    if "load_weights" in config:
        model.load_state_dict(torch.load(config.load_weights, map_location=f"cuda:{trainer.gpus[0]}"), strict=False)

    cfg_json = OmegaConf.to_container(config, resolve=True)
    dir_path = os.getcwd()
    with open(f"{dir_path}/config_all.json", "w") as fp:
        json.dump(cfg_json, fp)
    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    # if not config.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]

import argparse
import json

import hydra
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from tqdm import tqdm


def load_json_config(base_path, experiment_folder, model_name, fold):
    path = str(Path(base_path) / experiment_folder / model_name / fold / 'config_all.json')
    with open(path, "r") as fp:
        cfg = json.load(fp)
    omega_cfg = OmegaConf.create(cfg)
    return omega_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_folder")
    parser.add_argument("--backbone_name")
    parser.add_argument("--folds", type=str)
    parser.add_argument("--output_folder", default="data/predictions")
    args = parser.parse_args()

    BASE_PATH = "./logs/runs"
    USE_CKPTS = 2
    DEVICE = 0
    val_preds_list, val_labels_list, val_ids_list, test_preds_list, test_ids_list = [], [], [], [], []
    for fold in args.folds.split(","):
        config = load_json_config(BASE_PATH, args.experiment_folder, args.backbone_name, fold)
        print(config)
        config.trainer.gpus = [DEVICE]
        config.datamodule.val_fold = int(fold)
        search_path = Path(BASE_PATH) / args.experiment_folder / args.backbone_name / fold / "checkpoints"
        checkpoints = list((search_path).glob("*.ckpt"))
        checkpoints = sorted(checkpoints, key=lambda x: float(x.stem.split("=")[-1]), reverse=True)[:USE_CKPTS]

        # Init Lightning datamodule
        print(f"Instantiating datamodule <{config.datamodule._target_}>")
        # config.datamodule.batch_size=8
        datamodule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup()

        # Init Lightning model
        print(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        # Init Lightning trainer
        print(f"Instantiating trainer <{config.trainer._target_}>")
        config.trainer.precision = 32
        trainer = hydra.utils.instantiate(
            config.trainer
        )
        trainer.logger = None

        # Test the model
        val_preds_fold, test_preds_fold = [], []
        print("Starting testing!")
        for checkpoint in checkpoints:
            print(f"Checking for {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location=f"cuda:{DEVICE}")["state_dict"])
            val_preds = trainer.test(model, datamodule.val_dataloader())[0]['outputs']
            test_preds = trainer.test(model, datamodule.test_dataloader())[0]['outputs']
            val_labels = datamodule.val_data.labels
            val_ids = datamodule.val_data.ids
            test_ids_list = datamodule.test_data.ids
            val_preds_fold.append(val_preds)
            test_preds_fold.append(test_preds)
        val_preds_fold = np.mean(val_preds_fold, 0)
        test_preds_fold = np.mean(test_preds_fold, 0)
        val_preds_list.extend(val_preds_fold)
        val_labels_list.extend(val_labels)
        val_ids_list.extend(val_ids)
        test_preds_list.append(test_preds_fold)
    val_preds_list, val_labels_list = np.array(val_preds_list), np.array(val_labels_list)
    test_preds_list = np.mean(test_preds_list, 0)
    val_df = pd.DataFrame({'id': val_ids_list, 'target': val_labels_list, 'preds': val_preds_list})
    test_df = pd.DataFrame({'id': test_ids_list, 'target': test_preds_list})
    output_path = Path(args.output_folder) / args.experiment_folder / args.backbone_name
    output_path.mkdir(exist_ok=True, parents=True)
    print("ROC - AUC Score :", roc_auc_score(val_df.target.values, val_df.preds.values))
    val_df.to_csv(output_path / 'oof_preds.csv', index=False)
    test_df.to_csv(output_path / 'test_preds.csv', index=False)

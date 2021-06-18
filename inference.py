import argparse

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from tqdm import tqdm


def valid_preds(model, datamodule, device):
    model.eval()
    model.to(device)
    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(datamodule.val_dataloader()):
            x, y = batch['im'], batch['label']
            x, y = x.to(device), y.to(device)
            logits = model(x)
            yhat = torch.sigmoid(logits)
            preds.append(yhat)
            targets.append(y)
    preds = torch.cat(preds, 0).cpu().numpy()[:, 0]
    targets = torch.cat(targets, 0).cpu().numpy()
    print("Valid score: ", roc_auc_score(targets, preds))
    return preds


def test_preds(model, datamodule, device):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(datamodule.test_dataloader())):
            x, y = batch['im'], batch['label']
            x, y = x.to(device), y.to(device)
            logits = model(x)
            yhat = torch.sigmoid(logits)
            preds.append(yhat)
    preds = torch.cat(preds, 0).cpu().numpy()[:, 0]
    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--type")
    parser.add_argument("--model_path")
    parser.add_argument("--cfg")
    parser.add_argument("--val_fold", type=int)
    args = parser.parse_args()
    ITYPE = args.type
    model_ckpt = args.model_path  # "./logs/runs/vs_effv2s_mixup_augs_obsp8/efficientnetv2_rw_s/0/checkpoints/epoch=17-val/auc=0.9893.ckpt"
    cfg = OmegaConf.load(args.cfg)
    cfg.data_dir = "./data"
    #cfg.datamodule.power = 0.4
    cfg.datamodule.val_fold = args.val_fold
    print(cfg)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(model_ckpt, map_location=args.device)["state_dict"])
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()
    if ITYPE == "OOF":
        preds = valid_preds(model, datamodule, device=args.device)
        ids = datamodule.val_data.ids
        df = pd.DataFrame({"id": ids, "target": preds})
        df.to_csv(f"data/oof_{cfg.experiment_name}_{cfg.datamodule.val_fold}.csv", index=False)
    else:
        preds = test_preds(model, datamodule, device=args.device)
        ids = datamodule.test_data.ids
        df = pd.DataFrame({"id": ids, "target": preds})
        df.to_csv(f"data/sub_{cfg.experiment_name}_{cfg.datamodule.val_fold}.csv", index=False)

    
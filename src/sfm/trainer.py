import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import os

import sklearn
from sklearn.datasets import make_moons, make_circles

from torch import Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import odeint

from sfm.loggingwrapper import get_logger
from sfm.flowmodel import CNF, MLPwithTimeEmbedding, TargetConditionalFlowMatchingLoss

def get_dataset(n_samples: int, dataset: str = "moons", datanoise: float = 0.05, **kwargs):
    if dataset == "moons":
        # data is R^2 -> R
        data, _ = make_moons(n_samples, noise=datanoise)
        data = torch.from_numpy(data).float()
    elif dataset == "circles":
        data, _ = make_circles(n_samples, noise=datanoise)
        data = torch.from_numpy(data).float()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return data

def get_lr_schedule(optimizer, cfg: Dict):
    if cfg['lr_schedule'] == 'constant':
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=cfg['n_trainsteps'],
        )
    # TODO: they don't really work
    elif cfg['lr_schedule'] == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=cfg['optim']['lr_min'],
            total_iters=cfg['n_trainsteps'],
        )
    elif cfg['lr_schedule'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['n_trainsteps'] // 2,
            eta_min=cfg['optim']['lr_min'],
        )
    else:
        raise ValueError(f"Unknown lr schedule: {cfg['lr_schedule']}")

class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cfg = self.init_logging(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MLPwithTimeEmbedding(**cfg['model'])
        self.flow = CNF(model=model, source=cfg['source'])
        self.data: torch.Tensor = get_dataset(
            # 20% extra for validation set
            int(cfg['n_samples'] * 1.2),
            cfg['dataset'],
            cfg['datanoise'],
        )
        self.data_train = self.data[: cfg['n_samples']]
        self.data_val = self.data[cfg['n_samples'] :]

        self.step = 0

    def train(self) -> List[float]:
        # Training
        loss_fn = TargetConditionalFlowMatchingLoss(self.flow)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.cfg['optim']['lr'])
        lr_schedule = get_lr_schedule(optimizer, self.cfg)

        # Training loop for flow matching
        losses = []
        log_probs = []
        for trainstep in tqdm(range(self.cfg['n_trainsteps']), ncols=88):
            # Randomly select a batch of data
            subset = torch.randint(0, len(self.data_train), (self.cfg['batch_size'],))
            x = self.data_train[subset]

            loss = loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            losses.append(loss.item())

            if trainstep % self.cfg['logfreq'] == 0:
                self.logger.log({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']}, step=self.step, split="train")

            if trainstep % self.cfg['evalfreq'] == 0:
                gensamples, log_p = self.evaluate()
                self.logger.log({"log_p": log_p.mean()}, step=self.step, split="val")
                log_probs.append(log_p.mean())
                plot_data(x=gensamples, cfg=self.cfg, step=self.step)

            self.step += 1

        return losses

    def evaluate(self, eval_samples: int = None) -> Tuple[Tensor, Tensor]:
        # Generate samples from the flow
        with torch.no_grad():
            z = self.flow.source.sample(eval_samples or self.cfg['n_samples'])
            x = self.flow.decode(z) # [B, D]

        # Log-likelihood of true unseen data under the flow
        with torch.no_grad():
            log_p = self.flow.log_prob(
                self.data_val[: self.cfg['batch_size']] # [B, D]
            ) 

        return x, log_p

    def init_logging(self, cfg: Dict):
        self.logger = get_logger(cfg)
        # Create a directory for logging
        logdir = f"logs/{cfg['dataset']}_{cfg['source']['type']}"
        os.makedirs(logdir, exist_ok=True)
        cfg['logdir'] = logdir
        return cfg

    def finalize(self):
        self.logger.stop()


def plot_data(x, cfg, step: int, folder: str = None, fname: str = None):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    assert x.shape[1] == cfg['datadim'], f"Expected (...,{cfg['datadim']}) dimensions, got {x.shape}"

    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.hist2d(*x.T, bins=64)
    if fname is None:
        fname = f"{cfg['dataset']}_{cfg['source']['type']}_s{step}.png"
    if folder is None:
        folder = cfg['logdir']
    fname = folder + "/" + fname
    plt.savefig(fname)
    print(f"Saved data plot to\n {fname}")


def plot_loss(losses, cfg, folder: str = None, fname: str = None):
    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if fname is None:
        fname = f"{cfg['dataset']}_{cfg['source']['type']}_loss.png"
    if folder is None:
        folder = cfg['logdir']
    fname = folder + "/" + fname
    plt.savefig(fname)
    print(f"Saved loss plot to\n {fname}")
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
from sfm.flowmodel import CNF, FlowMatchingLoss

"""
In score-based generative modeling, it is standard to set t=0 as the data (noiseless) extremity and t=1 as the noise extremity. 
In the Flow Matching paper the authors reverse it, but we use the score-based convention in this implementation.
"""


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


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.cfg = self.init_logging(cfg)

        self.flow = CNF(cfg["datadim"], hidden_features=[64] * 3, source=cfg["source"])
        self.data: torch.Tensor = get_dataset(
            # 20% extra for validation set
            int(cfg["n_samples"] * 1.2),
            cfg["dataset"],
            cfg["datanoise"],
        )
        self.data_train = self.data[: cfg["n_samples"]]
        self.data_val = self.data[cfg["n_samples"] :]

        self.step = 0

    def train(self) -> List[float]:
        # Training
        loss_fn = FlowMatchingLoss(self.flow)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.cfg["optim"]["lr"])

        # Training loop for flow matching
        losses = []
        log_probs = []
        for epoch in tqdm(range(self.cfg["n_samples"]), ncols=88):
            # Randomly select a batch of data
            subset = torch.randint(0, len(self.data_train), (self.cfg["batch_size"],))
            x = self.data_train[subset]

            loss = loss_fn(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % self.cfg["logfreq"] == 0:
                self.logger.log({"loss": loss.item()}, step=self.step, split="train")

            if epoch % self.cfg["evalfreq"] == 0:
                gensamples, log_p = self.evaluate()
                self.logger.log({"log_p": log_p.mean()}, step=self.step, split="val")
                log_probs.append(log_p.mean())
                plot_data(x=gensamples, cfg=self.cfg, step=self.step)

            self.step += 1

        return losses

    def evaluate(self) -> Tuple[Tensor, Tensor]:
        # Generate samples from the flow
        with torch.no_grad():
            z = self.flow.source.sample(self.cfg["n_samples"])
            x = self.flow.decode(z)

        # Log-likelihood of true unseen data under the flow
        with torch.no_grad():
            log_p = self.flow.log_prob(self.data_val[: self.cfg["batch_size"]])

        return x, log_p

    def init_logging(self, cfg: Dict):
        self.logger = get_logger(cfg)
        # Create a directory for logging
        logdir = f"logs/{cfg['dataset']}_{cfg['source']['type']}"
        os.makedirs(logdir, exist_ok=True)
        cfg["logdir"] = logdir
        return cfg

    def finalize(self):
        self.logger.stop()


def plot_data(x, cfg, step: int, folder: str = None, fname: str = None):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    assert x.shape[1] == cfg["datadim"], f"Expected (...,{cfg['datadim']}) dimensions, got {x.shape}"

    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.hist2d(*x.T, bins=64)
    if fname is None:
        fname = f"{cfg['dataset']}_{cfg['source']['type']}_s{step}.png"
    if folder is None:
        folder = cfg["logdir"]
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
        folder = cfg["logdir"]
    fname = folder + "/" + fname
    plt.savefig(fname)
    print(f"Saved loss plot to\n {fname}")


@hydra.main(config_name="base", config_path="../src/sfm/config", version_base="1.3")
def run_with_hydra(args: DictConfig) -> None:
    # to dict
    cfg = OmegaConf.structured(OmegaConf.to_yaml(args))

    # Training
    trainer = Trainer(cfg)
    losses = trainer.train()

    # Final evaluation
    gensamples, log_p = trainer.evaluate()
    trainer.logger.log({"log_p": log_p.mean()}, step=trainer.step, split="val")

    # Plot the generated samples and the loss
    plot_data(x=gensamples, cfg=cfg, step=trainer.step)
    plot_loss(losses=losses, cfg=cfg)

    # Plot the training and validation distributions
    plot_data(x=trainer.data_train, cfg=cfg, step=0, folder="plots/data", fname=f"{cfg['dataset']}_train.png")
    plot_data(x=trainer.data_val, cfg=cfg, step=0, folder="plots/data", fname=f"{cfg['dataset']}_val.png")

    print(f"Log probability: {log_p.mean():.3f} ± {log_p.std():.3f}")
    print(f"Loss after {cfg['n_samples']} n_samples: {losses[-1]:.3f}")

    trainer.finalize()


if __name__ == "__main__":
    run_with_hydra()

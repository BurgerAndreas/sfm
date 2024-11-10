import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import os

import torchdyn

import sklearn
from sklearn.datasets import make_moons, make_circles

from torch import Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import odeint

from sfm.loggingwrapper import get_logger
from sfm.flowmodel import (
    ContNormFlow,
    MLPwithTimeEmbedding,
    LipmanFMLoss,
    LipmanTCFMLoss,
    OTCFMLoss,
    CFMLoss,
    torchdyn_wrapper,
    NeuralODEWrapper,
)

from sfm.distributions import get_source_distribution

def norm_data_01(data):
    # make sure data is inside [0, 1]
    # using max and min along each dimension
    # shape is (n_samples, 2)
    data[:, 0] = (data[:, 0] - data[:, 0].min()) / (data[:, 0].max() - data[:, 0].min())
    data[:, 1] = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min())
    return data

def norm_data_pm1(data):
    # make sure data is inside [-1, 1]
    # using max and min along each dimension
    # shape is (n_samples, 2)
    norm_data_01(data)
    data[:, 0] = data[:, 0] * 2 - 1
    data[:, 1] = data[:, 1] * 2 - 1
    return data

def get_dataset(n_samples: int, trgt: str = "moons", noise: float = 0.05, **kwargs):
    # https://scikit-learn.org/dev/api/sklearn.datasets.html#sample-generators
    if trgt == "moons":
        data, _ = make_moons(n_samples, noise=noise)
        data = torch.from_numpy(data).float()
    elif trgt == "circles":
        data, _ = make_circles(n_samples, noise=noise)
        data = torch.from_numpy(data).float()
    else:
        raise ValueError(f"Unknown dataset: {trgt}")
    norm_data_01(data)
    return data


def get_lr_schedule(optimizer, cfg: Dict):
    if cfg["lr_schedule"] == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=cfg["n_trainsteps"],
        )
    # TODO: they don't really work
    elif cfg["lr_schedule"] == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=cfg["optim"]["lr_min"],
            total_iters=cfg["n_trainsteps"],
        )
    elif cfg["lr_schedule"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["n_trainsteps"] // 2,
            eta_min=cfg["optim"]["lr_min"],
        )
    else:
        raise ValueError(f"Unknown lr schedule: {cfg['lr_schedule']}")


class Trainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.fnamebase = f"{cfg['data']['trgt']}_{cfg['source']['trgt']}_{cfg['fmloss']}_{cfg['n_ode']}"

        self.cfg = self.init_logging(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cfg["force_cpu"] else "cpu")
        torch.set_default_device(self.device)

        model = MLPwithTimeEmbedding(**cfg["model"], device=self.device).to(self.device)
        if cfg["n_ode"] == "zuko":
            self.flow = ContNormFlow(model=model, fmtime=cfg["fmtime"])
        elif cfg["n_ode"] == "torchdyn":
            self.flow = NeuralODEWrapper(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4).to(
                self.device
            )
        else:
            raise ValueError(f"Unknown neural ode type: {cfg['n_ode']}")

        self.sourcedist = get_source_distribution(**cfg["source"], device=self.device)

        # currently all on cpu
        self.data: torch.Tensor = get_dataset(
            # 20% extra for validation set
            int(cfg["n_samples"] * 1.2),
            **cfg["data"],
        )
        self.data_train = self.data[: cfg["n_samples"]]
        self.data_val = self.data[cfg["n_samples"] :]

        self.step = 0

    def train(self) -> List[float]:
        # Training
        if self.cfg["fmloss"] == "lipman":
            loss_fn = LipmanFMLoss(self.flow).to(self.device)
        elif self.cfg["fmloss"] == "lipmantcfm":
            loss_fn = LipmanTCFMLoss(self.flow).to(self.device)
        elif self.cfg["fmloss"] == "cfm":
            loss_fn = CFMLoss(self.flow).to(self.device)
        elif self.cfg["fmloss"] == "otcfm":
            loss_fn = OTCFMLoss(self.flow).to(self.device)
        else:
            raise ValueError(f"Unknown loss function: {self.cfg['fmloss']}")
        # loss function determines the time convention
        if self.cfg["fmtime"] is None:
            self.cfg["fmtime"] = loss_fn.fmtime
            self.flow.set_fmtime(self.cfg["fmtime"])

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.cfg["optim"]["lr"])
        lr_schedule = get_lr_schedule(optimizer, self.cfg)

        # Training loop for flow matching
        losses = []
        log_probs = []
        for trainstep in tqdm(range(self.cfg["n_trainsteps"]), ncols=44):
            # Randomly select a batch of data = samples from the data distribution
            subset = torch.randint(0, len(self.data_train), (self.cfg["batch_size"],), device=self.data_train.device)
            targets = self.data_train[subset].to(self.device)
            # sample from the source distribution
            sources = self.sourcedist.sample(self.cfg["batch_size"]).to(self.device)

            loss = loss_fn(sources=sources, targets=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            losses.append(loss.item())

            if trainstep % self.cfg["logfreq"] == 0:
                self.logger.log(
                    {"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=self.step, split="train"
                )

            if trainstep % self.cfg["evalfreq"] == 0:
                gensamples, log_p = self.evaluate()
                # log_p tends to be contain NaNs if sample is outside of support of source distribution
                log_p = torch.nanmean(log_p)
                self.logger.log({"log_p": log_p}, step=self.step, split="val")
                log_probs.append(log_p.item())
                fname, fig = self.plot_data(x=gensamples, step=self.step)
                self.logger.log_img(fig, key="gen", step=self.step, split="val")
                tqdm.write(f"Saved data plot to\n {fname}")

            self.step += 1

        return losses, log_probs

    def evaluate(self, eval_samples: int = None) -> Tuple[Tensor, Tensor]:
        # Generate samples from the flow
        with torch.no_grad():
            sources = self.sourcedist.sample(eval_samples or self.cfg["n_samples"]).to(self.device)
            gen_targets = self.flow.decode(sources)  # [B, D]

        # Log-likelihood of true unseen data under the flow
        with torch.no_grad():
            log_p = self.flow.log_prob(
                targets=self.data_val[: self.cfg["batch_size"]].to(self.device),  # [B, D]
                sourcedist=self.sourcedist,
            )

        return gen_targets, log_p

    def init_logging(self, cfg: Dict):
        # logging to file
        logdir = f"logs/{self.fnamebase}"
        os.makedirs(logdir, exist_ok=True)
        cfg["logdir"] = logdir
        # logging to cloud service
        self.logger = get_logger(cfg)
        self.logger.logdir = logdir
        self.logdir = logdir
        return cfg
    
    def savetofile(self, data, fname: str):
        torch.save(data, f"{self.logdir}/{fname}.pt")

    def finalize(self):
        self.logger.stop()

    def plot_data(self, x, step: int, folder: str = None, fname: str = ""):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        assert x.shape[1] == self.cfg["datadim"], f"Expected (...,{self.cfg['datadim']}) dimensions, got {x.shape}"

        fig = plt.figure(figsize=(4.8, 4.8), dpi=150)
        plt.hist2d(*x.T, bins=64)
        fname += f"_s{step}.png"
        if folder is None:
            folder = self.logdir
        fname = folder + "/" + fname
        # plt.savefig() would close the figure
        fig.savefig(fname) 
        return fname, fig

    def plot_loss(self, losses, folder: str = None, fname: str = ""):
        fig = plt.figure(figsize=(4.8, 4.8), dpi=150)
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        fname += "_loss.png"
        if folder is None:
            folder = self.logdir
        fname = folder + "/" + fname
        # plt.savefig() would close the figure
        fig.savefig(fname)
        return fname, fig

# from Flow_matching_tutorial.ipynb
# First (CFM) and second part (OT) of the tutorial

import math
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import generate_moons
import torchdiffeq
import torchsde
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchdyn.core import DEFunc, NeuralODE
from torchdyn.datasets import generate_moons
from torchdyn.nn import Augmenter

from torchcfm.utils import torch_wrapper

import hydra
from omegaconf import DictConfig

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

import torchcfm.models.models as tcfm_models
from torchcfm.utils import sample_8gaussians, sample_moons, plot_trajectories, torch_wrapper

from torchcfm.optimal_transport import OTPlanSampler

from sfm.networks import get_model
from sfm.flowmodel import ContNormFlow
from sfm.distributions import get_source_distribution
from sfm.tcfmhelpers import sample_conditional_pt, compute_conditional_vector_field
from sfm.tcfmhelpers import CNF


def train_cfm(args: DictConfig):
    assert args.savedir not in ["", None, "None"], "savedir is required"
    os.makedirs(f"{args.savedir}/train", exist_ok=True)
    print(f"Saving to {args.savedir}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.force_cpu:
        device = torch.device("cpu")
    print("Using device:", device)

    ot_sampler = OTPlanSampler(method="exact")
    sigma = 0.1 # for flow matching
    dim = 2 # dimension of the data
    n_int_steps = 100 # number of integration steps
    tnoise = 0 # noise time
    tdata = 1 # data time

    sourcedist = get_source_distribution(**args.source)
    model = get_model(**args.model, dim=dim).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    flowmatching = ConditionalFlowMatcher(sigma=sigma)
    # Target FM (Lipman et al. 2023), only works with Gaussian source
    # FM = TargetConditionalFlowMatcher(sigma=sigma)
    # Exact OT CFM
    # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    ).to(device)

    # Initialize list to store losses
    losses = []
    logprobs_train = []
    
    start = time.time()
    for k in tqdm(range(args.n_trainsteps)):
        optimizer.zero_grad()

        # sample source distribution [B, D]
        # x0 = sample_8gaussians(args.batch_size).to(device)
        x0 = sourcedist.sample(args.batch_size).to(device)
        assert x0.shape == (args.batch_size, dim)
        # sample target distribution [B, D]
        x1 = sample_moons(args.batch_size).to(device)

        # Draw samples from OT plan
        # only difference between ConditionalFlowMatcher and ExactOptimalTransportConditionalFlowMatcher
        if args.use_ot:
            x0, x1 = ot_sampler.sample_plan(x0, x1)

        if args.use_slcf:
            # [B], [B, D], [B, D]
            t, xt, ut = flowmatching.sample_location_and_conditional_flow(x0, x1)
        else:
            t = torch.rand(x0.shape[0], device=device).type_as(x0)
            xt = sample_conditional_pt(x0, x1, t, sigma=0.01, device=device)
            ut = compute_conditional_vector_field(x0, x1, device=device)

        # [B, D+1] -> [B, D]
        vt = model(
            torch.cat([xt, t[:, None]], dim=-1) # [B,D], [B] -> [B,D+1]
        )
        # vt = model(t, xt, y=None)

        loss = torch.nanmean((vt - ut) ** 2)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store loss value
        losses.append([k, loss.item()])

        # evaluate model
        if ((k + 1) % args.evalfreq == 0) or (k == 0):
            end = time.time()
            tqdm.write(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
            # generate samples and plot trajectories
            with torch.no_grad():
                # [T, B, D]
                traj = node.trajectory(
                    # x=sample_8gaussians(args.eval_batch_size).to(device),
                    x=sourcedist.sample(args.eval_batch_size).to(device),
                    t_span=torch.linspace(tnoise, tdata, n_int_steps, device=device),
                )
                fig = plot_trajectories(traj.cpu().numpy())
                fig.savefig(f"{args.savedir}/train/traj_{k}.png")
                plt.close(fig)
                print(f"Saved trajectory to {args.savedir}/train/traj_{k}.png")
            
            # compute log-likelihood of test set
            # starting from target points, integrate backwards from t=1 to t=0 to get the source points
            # compute the log-likelihood using the source distribution and the trace term
            cnf = DEFunc(CNF(model))
            nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
            cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
            # with torch.no_grad():
            x1 = sample_moons(args.eval_batch_size).to(device).requires_grad_()
            # integrate backwards from t=1 (tdata) to t=0 (tnoise)
            aug_traj = (
                cnf_model[1]
                .to(device)
                .trajectory(
                    Augmenter(1, 1)(x1).to(device),
                    t_span=torch.linspace(start=tdata, end=tnoise, steps=n_int_steps, device=device),
                )
            )[-1].cpu()
            # Compute log probabilities
            log_probs = sourcedist.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
            logprobs_train.append([k, log_probs.nanmean().item()])
            print(f"Log-likelihood of test set: {log_probs.nanmean().item():0.3f}")

    # save model
    torch.save(model.state_dict(), f"{args.savedir}/{args.cpname}.pth")
    print(f"Saved model to {args.savedir}/{args.cpname}.pth")
    
    # save losses
    losses = np.array(losses)
    np.save(f"{args.savedir}/losses.npy", losses)
    print(f"Saved losses to {args.savedir}/losses.npy")
    logprobs_train = np.array(logprobs_train)
    np.save(f"{args.savedir}/logprobs_train.npy", logprobs_train)
    print(f"Saved logprobs_train to {args.savedir}/logprobs.npy")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    train_cfm(args)
    
if __name__ == "__main__":
    hydra_wrapper()
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
import torchdiffeq
import torchsde
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

import torchdyn.core as tdyn
from torchdyn.nn import Augmenter

from torchcfm.utils import torch_wrapper

import hydra
from omegaconf import DictConfig

from torchcfm.utils import torch_wrapper

from sfm.networks import get_model
from sfm.flowmodel import ContNormFlow
from sfm.distributions import get_source_distribution
from sfm.datasets import sample_dataset, get_dataset
from sfm.tcfmhelpers import sample_conditional_pt, compute_conditional_vector_field
from sfm.tcfmhelpers import CNF, plot_trajectories
from sfm.plotstyle import set_seaborn_style, set_style_after
from sfm.flowmatching import get_flowmatching

def get_trajplot_name(args: DictConfig, k: int):
    # traj_{k}_steps{nsteps}.png
    return f"{args.savedir}/traj_{k}_is{args.eval_integration_steps}.png"

def get_logprob_name(args: DictConfig):
    return f"{args.savedir}/logprobs_is{args.eval_integration_steps}.npy"

def get_loss_name(args: DictConfig):
    return f"{args.savedir}/losses.npy"

def get_model_name(args: DictConfig):
    return f"{args.savedir}/{args.cpname}.pth"

def eval_traj(args: DictConfig, model, node, device, sourcedist, tnoise, tdata, nintsteps, k):
    with torch.no_grad():
        if args.classcond:
            # y0: torch.Size([B, 1, 28, 28])
            y0 = sourcedist.sample(args.eval_batch_size).to(device)
            y0 = y0.view(y0.shape[0], 1, 28, 28)
            generated_class_list = torch.arange(10, device=device).repeat(args.eval_batch_size // 10 + 1) # [100]
            generated_class_list = generated_class_list[:args.eval_batch_size]
            traj = torchdiffeq.odeint(
                func=lambda t, x: model.forward(t, x, generated_class_list),
                # y0=torch.randn(100, 1, 28, 28, device=device),
                y0=y0,
                t=torch.linspace(tnoise, tdata, 2, device=device), # [2]
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
            # MNIST: (2, B, 1, 28, 28) 
            # only plot the last time step
            nrows = 3
            _gen = traj[-1][:nrows**2]
            # choose first 9 samples, and plot them in a 3x3 grid
            grid = make_grid(
                _gen.view([-1, *(1, 28, 28)]).clip(-1, 1), 
                value_range=(-1, 1), padding=0, nrow=nrows
            )
            img = ToPILImage()(grid)
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout(pad=0.0)
            plt.savefig(get_trajplot_name(args, k))
            plt.close()
            print(f"Saved trajectory to {get_trajplot_name(args, k)}")
        else:
            # [T, B, D]
            traj = node.trajectory(
                # x=sample_8gaussians(args.eval_batch_size).to(device),
                x=sourcedist.sample(args.eval_batch_size).to(device),
                t_span=torch.linspace(tnoise, tdata, nintsteps, device=device),
            )
            # [intsteps, B, D])
            fig = plot_trajectories(traj.cpu().numpy())
            fig.savefig(get_trajplot_name(args, k))
            plt.close(fig)
            print(f"Saved trajectory to {get_trajplot_name(args, k)}")
    return traj

def eval_logprob(args: DictConfig, model, device, sourcedist, trgtdist, tnoise, tdata, nintsteps, logprobs_train, k):
    # starting from target points, integrate backwards from t=1 to t=0 to get the source points
    # compute the log-likelihood using the source distribution and the trace term
    if args.classcond:
        logprobs_train = None # TODO: implement
    else:
        nde = tdyn.NeuralODE(tdyn.DEFunc(CNF(model)), solver="euler", sensitivity="adjoint")
        cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
        # with torch.no_grad():
        # x1 = sample_moons(args.eval_batch_size).to(device).requires_grad_()
        x1 = trgtdist.sample(args.eval_batch_size).to(device).requires_grad_()
        # integrate backwards from t=1 (tdata) to t=0 (tnoise)
        aug_traj = (
            cnf_model[1]
            .to(device)
            .trajectory(
                x=Augmenter(1, 1)(x1).to(device),
                t_span=torch.linspace(start=tdata, end=tnoise, steps=nintsteps, device=device),
            )
        )[-1] # [B, D+1]
        # Compute log probabilities
        # We can load the log probs later to plot the training progress
        # logprob: [B, D] -> [B]
        log_probs = sourcedist.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
        logprobs_train.append([k, log_probs.nanmean().item()])
        print(f"Log-likelihood of test set: {log_probs.nanmean().item():0.3f}")
    return logprobs_train

def train_cfm(args: DictConfig):
    print("\n" + "-"*80 + f"\nTraining CFM for {args.runname}\n")
    
    assert args.savedir not in ["", None, "None"], "savedir is required"
    os.makedirs(f"{args.savedir}/train", exist_ok=True)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.force_cpu:
        device = torch.device("cpu")
    print("Using device:", device)
    
    if os.path.exists(get_model_name(args)):
        if args.force_retrain:
            print(f"Force retraining: {get_model_name(args)}")
        else:
            print(f"Model already exists: {get_model_name(args)}")
            return

    # ot_sampler = OTPlanSampler(method="exact")
    nintsteps = args.eval_integration_steps 
    tnoise = 0 # noise time
    tdata = 1 # data time

    trgtdist = get_dataset(**args.data)
    sourcedist = get_source_distribution(**args.source, trgtdist=trgtdist, device=device)
    
    if args.source.data_dim == 2:
        # plot the target distribution
        # plot target distribution samples
        noise = sourcedist.sample(1000).cpu().numpy() #
        samples = trgtdist.sample(1000).cpu().numpy()
        set_seaborn_style()
        fig = plt.figure(figsize=(6,6))
        plt.scatter(samples[:,0], samples[:,1], alpha=0.5, s=2)
        set_style_after(ax=fig.gca())
        plt.savefig(f"{args.savedir}/targetdist.png")
        print(f"Saved {args.savedir}/targetdist.png")
        plt.scatter(noise[:,0], noise[:,1], alpha=0.5, s=2)
        plt.savefig(f"{args.savedir}/srctrgt.png")
        print(f"Saved {args.savedir}/srctrgt.png")
        plt.close()
    
    model = get_model(**args.model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    flowmatching = get_flowmatching(args)

    node = tdyn.NeuralODE(
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
        
        # sample target distribution [B, D]
        # x1 = sample_moons(args.batch_size).to(device)
        x1 = trgtdist.sample(args.batch_size)
        # print(f"data x:  {x1[:, 0].min().item():0.3f}, {x1[:, 0].max().item():0.3f}")
        # print(f"data y:  {x1[:, 1].min().item():0.3f}, {x1[:, 1].max().item():0.3f}")
        # print(f"noise x: {x0[:, 0].min().item():0.3f}, {x0[:, 0].max().item():0.3f}")
        # print(f"noise y: {x0[:, 1].min().item():0.3f}, {x0[:, 1].max().item():0.3f}")
        
        # assert x0.max() <= 1.5 and x0.min() >= 0, "x0 is out of bounds"
        # assert x1.max() <= 1.5 and x1.min() >= 0, "x1 is out of bounds"

        if args.classcond:
            data = x1
            x1 = data[0].to(device)
            y = data[1].to(device)  # class labels
            assert x1.shape == x0.shape, \
                f"x1 and x0 must have the same shape but got {x1.shape} and {x0.shape}"
            # reshape x: [B,D] -> [B,1,28,28]
            x1 = x1.view(x1.shape[0], 1, 28, 28)
            x0 = x0.view(x0.shape[0], 1, 28, 28)
            if args.use_ot:
                # same as
                # x0, x1, y0, y1 = ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
                # t, xt, ut = ConditionalFlowMatcher.sample_location_and_conditional_flow(x0, x1, t, False)
                # return t, xt, ut, y0, y1
                # [B], [B, 1, 28, 28], [B, 1, 28, 28], None, [B]
                t, xt, ut, _, y1 = flowmatching.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
                vt = model(t, xt, y1)
            else:
                t, xt, ut = flowmatching.sample_location_and_conditional_flow(x0, x1)
                vt = model(t, xt, y)
        else:
            x1 = x1.to(device)
            # Draw samples from OT plan
            # only difference between ConditionalFlowMatcher and ExactOptimalTransportConditionalFlowMatcher
            # if args.use_ot:
            #     x0, x1 = ot_sampler.sample_plan(x0, x1)

            # [B], [B, D], [B, D]
            t, xt, ut = flowmatching.sample_location_and_conditional_flow(x0, x1)

            # [B, D+1] -> [B, D]
            vt = model(
                torch.cat([xt, t[:, None]], dim=-1) # [B,D], [B] -> [B,D+1]
            )

        loss = torch.mean((vt - ut) ** 2) * args.loss_scale

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Store loss value
        losses.append([k, loss.item()])

        # evaluate model
        # this part depends on the number of integration steps
        if ((k + 1) % args.evalfreq == 0) or (k == 0):
            end = time.time()
            tqdm.write(f"{k+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
            start = end
            # generate samples and plot trajectories
            eval_traj(args, model, node, device, sourcedist, tnoise, tdata, nintsteps, k)
            
            # compute log-likelihood of test set
            logprobs_train = eval_logprob(args, model, device, sourcedist, trgtdist, tnoise, tdata, nintsteps, logprobs_train, k)

    # save model
    torch.save(model.state_dict(), get_model_name(args))
    print(f"Saved model to {get_model_name(args)}")
    
    # save losses
    losses = np.array(losses)
    np.save(get_loss_name(args), losses)
    print(f"Saved losses to {get_loss_name(args)}")
    if logprobs_train is not None:
        logprobs_train = np.array(logprobs_train)
        np.save(get_logprob_name(args), logprobs_train)
        print(f"Saved logprobs_train to {get_logprob_name(args)}")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    train_cfm(args)
    
if __name__ == "__main__":
    hydra_wrapper()
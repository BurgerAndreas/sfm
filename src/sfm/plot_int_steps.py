# load a pretrained model
# for different integration steps:
# 1. plot the generated samples
# 2. plot the log-likelihood of validation samples

import math
import os
import time
import hydra

# import imageio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ot as pot
import torch
from torch import Tensor
import torchdyn
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.nn import Augmenter
from omegaconf import DictConfig
from tqdm import tqdm

from torchcfm.models.models import MLP, GradModel
from torchcfm.utils import torch_wrapper

from sfm.distributions import get_source_distribution
from sfm.tcfmhelpers import CNF
from sfm.plotstyle import _cscheme, set_seaborn_style
from sfm.networks import get_model
from sfm.datasets import get_dataset


def plot_integration_steps(args: DictConfig) -> None:
    print(f"Plotting integration steps for {args.runname}\n")
    
    limmin = args.plim[0]
    limmax = args.plim[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    
    # folder with temporary trajectory plots
    tempdir = f"{args.savedir}/integrationsteps"
    os.makedirs(tempdir, exist_ok=True)
    
    tplotname = "logprob_intsteps"
    combinedplotname = f"{tplotname}_{args['source']['trgt']}-to-{args.data['trgt']}"
    combinedplotname += f"-ot" if args.use_ot else ""
    
    n_models = 1
    n_samples = args.plot_batch_size
    max_integration_steps = args.plot_integration_steps
    n_integration_steps = 5
    n_img = 5 # number of images of the inference trajectory
    tdata = 1
    tnoise = 0
    
    model = get_model(**args.model).to(device)
    cp = torch.load(os.path.join(args.savedir, args.cpname + ".pth"), weights_only=True)
    model.load_state_dict(cp)
    
    sourcedist = get_source_distribution(**args.source)

    # sample noise
    # sample = sample_8gaussians(n_samples) # [n_samples, 2]
    sample = sourcedist.sample(n_samples) # [n_samples, 2]
    assert sample.shape == (n_samples, 2), f"sample.shape: {sample.shape}"

    trgtdist = get_dataset(**args.data)
    
    # plotting stuff for log-prob plot
    points = 100j
    points_real = 100
    Y, X = np.mgrid[limmin:limmax:points, limmin:limmax:points] # [points, points]
    gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32) # [points**2, 2]
    # quiver plot
    points_small = 20j
    points_real_small = 20
    Y_small, X_small = np.mgrid[limmin:limmax:points_small, limmin:limmax:points_small] # [points_small, points_small]
    gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
        torch.float32
    )
    
    logprobs_intsteps = []
    
    # integrate up to 1 / starting from 1 using intsteps steps
    # the better the flow the faster / fewer steps still give good results
    # will include 0 and max
    intsteps_list = torch.linspace(0, max_integration_steps, n_integration_steps)
    intsteps_list = [int(intsteps) for intsteps in intsteps_list]
    intsteps_list = [intsteps if intsteps != 1 else 2 for intsteps in intsteps_list]
    print(f"intsteps_list: {intsteps_list}")
    for intsteps in intsteps_list: 
        
        ### calculate log-likelihood of sample
        if args.classcond:
            pass
        else:
            cnf = DEFunc(CNF(model))
            nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
            cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
            # with torch.no_grad():
            # x1 = sample_moons(args.eval_batch_size).to(device).requires_grad_()
            x1 = trgtdist.sample(args.eval_batch_size).to(device).requires_grad_()
            if intsteps > 0:
                # if intsteps == 1:
                #     print("Warning: setting intsteps=2 because intsteps=1 is not allowed")
                #     intsteps = 2
                # integrate backwards from t=1 (tdata) to t=0 (tnoise)
                aug_traj = (
                cnf_model[1]
                .to(device)
                .trajectory(
                    x=Augmenter(1, 1)(x1).to(device),
                    t_span=torch.linspace(start=tdata, end=tnoise, steps=intsteps, device=device),
                    )
                )[-1].cpu()
                # Compute log probabilities
                # We can load the log probs later to plot the training progress
                log_probs = sourcedist.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
            else:
                log_probs = sourcedist.log_prob(x1)
            logprobs_intsteps.append([intsteps, log_probs.nanmean().item()])
            print(f"Log-likelihood intsteps={intsteps}: {log_probs.nanmean().item():0.3f}")

        ### trajectory plot?
        # src/sfm/plot_cfm_gif.py
        
        ### log-probability / density plot
        fig, axis = plt.subplots(1, 1, figsize=(6, 6))
        cnf = DEFunc(CNF(model))
        nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
        cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
        with torch.no_grad():
            if intsteps > 0:
                # integrate backwards from t to 0
                aug_traj = (
                    cnf_model[1]
                    .to(device)
                    .trajectory(
                        Augmenter(1, 1)(gridpoints).to(device),
                        t_span=torch.linspace(start=1, end=0, steps=intsteps, device=device),
                    )
                )[-1].cpu()
                # log_probs = log_8gaussian_density(aug_traj[:, 1:]) - aug_traj[:, 0]
                log_probs = sourcedist.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
            else:
                # no integration, just evaluate at t=0
                # log_probs = log_8gaussian_density(gridpoints)
                log_probs = sourcedist.log_prob(gridpoints)
        assert log_probs.shape == (points_real**2,), f"log_probs.shape: {log_probs.shape}"
        # print(f"Log-prob of sample: {sourcedist.log_prob(sample).nanmean().item()}")
        # tqdm.write(f"Log-prob of sample under model: {log_probs.nanmean().item():0.2f}")
        log_probs = log_probs.reshape(Y.shape)
        
        # # Convert log probabilities to probabilities and normalize
        # probs = torch.exp(log_probs)
        # # Normalize to [0,1] for better visibility
        # probs = (probs - probs.min()) / (probs.max() - probs.min())
        
        ax = axis
        # viridis coolwarm BuPu PuRd magma inferno cividis prism ocean
        cmap = "viridis" 
        # cmap = sns.color_palette("Spectral", as_cmap=True)
        ax.pcolormesh(
            X, Y, torch.exp(log_probs), 
            # vmax=1, 
            cmap=cmap,
            # shading{'flat', 'nearest', 'gouraud', 'auto'}
            shading='gouraud', norm='linear'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(limmin, limmax)
        ax.set_ylim(limmin, limmax)
        # ax.set_title(f"{args.runname}", fontsize=30)
        # can't get rid of white border, so pad a bit to make it look cleaner
        plt.tight_layout(pad=0.1) 
        plt.savefig(f"{tempdir}/{tplotname}_{intsteps}.png", dpi=40)
        plt.close()
        
        
    # load all density plots and put side by side into one figure
    fignames = [f"{tempdir}/{tplotname}_{intsteps}.png" for intsteps in intsteps_list]
    images = [imageio.imread(fname) for fname in fignames]
    
    # create a new figure with subplots side by side
    n_plots = len(images)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    # plot each image
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        # remove border between plots
        # ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([]) 
        # torn off minor ticks
        ax.tick_params(axis='both', which='minor', length=0)
        # ax.set_title(f"T={ts[i]:0.2f}")
    
    plt.tight_layout(pad=0.0)
    fname = f"{args.savedir}/{combinedplotname}.png"
    plt.savefig(fname, dpi=40)
    plt.close()
    print(f"Saved integration steps to\n {fname}")
    
    # save intsteps_list for reference
    np.save(f"{args.savedir}/{combinedplotname}_intsteps_list.npy", intsteps_list)
    print(f"Saved intsteps_list to {args.savedir}/{combinedplotname}_intsteps_list.npy")
    
    # save logprobs_intsteps
    logprobs_intsteps = np.array(logprobs_intsteps)
    np.save(f"{args.savedir}/{combinedplotname}_logprobs_intsteps.npy", logprobs_intsteps)
    print(f"Saved logprobs_intsteps to {args.savedir}/{combinedplotname}_logprobs_intsteps.npy")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_integration_steps(args)


if __name__ == "__main__":
    hydra_wrapper()
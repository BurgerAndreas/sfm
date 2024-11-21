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

PLOT_DIR_SOURCE = "plots/sources"

def set_seaborn_style(*args, **kwargs):
    pass

def plot_inference_sidebyside(args: DictConfig) -> None:
    print(f"Plotting inference sidebyside for {args.runname}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    
    # folder with temporary trajectory plots
    tempdir = f"{args.savedir}/sidebyside"
    os.makedirs(tempdir, exist_ok=True)
    
    tplotname = "logprob"
    combinedplotname = f"{tplotname}_{args['source']['trgt']}-to-{args.data['trgt']}"
    
    n_models = 1
    n_samples = 1024
    n_t_span = 201
    n_t_gif = 5
    
    # plotting stuff
    # log-prob plot
    points = 100j
    points_real = 100
    Y, X = np.mgrid[0:1:points, 0:1:points] # [points, points]
    gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32) # [points**2, 2]
    # quiver plot
    points_small = 20j
    points_real_small = 20
    Y_small, X_small = np.mgrid[0:1:points_small, 0:1:points_small] # [points_small, points_small]
    gridpoints_small = torch.tensor(np.stack([X_small.flatten(), Y_small.flatten()], axis=1)).type(
        torch.float32
    )
    
    model = MLP(dim=2, time_varying=True)
    cp = torch.load(os.path.join(args.savedir, args.cpname + ".pth"), weights_only=True)
    model.load_state_dict(cp)
    
    sourcedist = get_source_distribution(**args.source)

    # sample noise
    # sample = sample_8gaussians(n_samples) # [n_samples, 2]
    sample = sourcedist.sample(n_samples) # [n_samples, 2]
    assert sample.shape == (n_samples, 2), f"sample.shape: {sample.shape}"
    
    os.makedirs(PLOT_DIR_SOURCE, exist_ok=True)
    
    # plot sample
    set_seaborn_style()
    plt.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_scatter.png"
    plt.savefig(fname, dpi=40)
    print(f"Saved sample to\n {fname}")
    plt.close()
    
    # plot sample as 2d histogram
    set_seaborn_style()
    cmap = "coolwarm"
    # cmap = sns.color_palette("Spectral", as_cmap=True)
    # plt.hist2d(
    #     sample[:, 0].numpy(), sample[:, 1].numpy(), bins=100, cmap=cmap
    # )
    sns.histplot(
        x=sample[:, 0].numpy(), y=sample[:, 1].numpy(), cmap=cmap, fill=True,
        bins=100
    )
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_hist2d.png"
    plt.savefig(fname, dpi=40)
    print(f"Saved hist2d to\n {fname}")
    plt.close()
    
    # KDE plot
    sns.kdeplot(
        x=sample[:, 0].numpy(), y=sample[:, 1].numpy(), cmap=cmap, fill=True
    )
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_kde.png"
    plt.savefig(fname, dpi=40)
    print(f"Saved kde to\n {fname}")
    plt.close()
    
    
    ts = torch.linspace(0, 1, n_t_gif) # [n_t_gif]
    
    # compute trajectory once, later pick time slices for gif
    nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler").to(device)
    # with torch.no_grad():
    # [n_t_gif, n_samples, 2]
    traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()
    set_seaborn_style()
    n_plots = 1
    for i, t in tqdm(enumerate(ts)):
        fig, axes = plt.subplots(n_plots, n_models, figsize=(6 * n_models, 6 * n_plots))
        axis = axes if n_models == 1 else axes[:, 0]
        
        ### log-probability / density plot
        cnf = DEFunc(CNF(model))
        nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
        cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
        with torch.no_grad():
            if t > 0:
                # integrate backwards from t to 0
                aug_traj = (
                    cnf_model[1]
                    .to(device)
                    .trajectory(
                        Augmenter(1, 1)(gridpoints).to(device),
                        t_span=torch.linspace(start=t, end=0, steps=n_t_span, device=device),
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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.set_title(f"{args.runname}", fontsize=30)
        # can't get rid of white border, so pad a bit to make it look cleaner
        plt.tight_layout(pad=0.1) 
        plt.savefig(f"{tempdir}/{tplotname}_{t:0.2f}.png", dpi=40)
        plt.close()

    # load all density plots and put side by side into one figure
    fignames = [f"{tempdir}/{tplotname}_{t:0.2f}.png" for t in ts]
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
    fname = f"{args.savedir}/{combinedplotname}_sidebyside.png"
    plt.savefig(fname, dpi=40)
    plt.close()
    print(f"Saved sidebyside to\n {fname}")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_inference_sidebyside(args)


if __name__ == "__main__":
    hydra_wrapper()
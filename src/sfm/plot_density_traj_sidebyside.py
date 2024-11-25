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
import torchdyn.core as tdyn
from torchdyn.nn import Augmenter
from omegaconf import DictConfig
from tqdm import tqdm

from torchcfm.models.models import MLP, GradModel
from torchcfm.utils import torch_wrapper

from sfm.distributions import get_source_distribution
from sfm.datasets import get_dataset
from sfm.networks import get_model
from sfm.tcfmhelpers import CNF
from sfm.plotstyle import _cscheme, set_seaborn_style

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import torchdiffeq

PLOT_DIR_SOURCE = "plots/sources"

# def set_seaborn_style(*args, **kwargs):
#     pass

def plot_samples(args: DictConfig, sample: Tensor) -> None:
    # plot sample
    set_seaborn_style()
    sample = sample.detach().cpu().numpy()
    plt.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_scatter.png"
    plt.savefig(fname, dpi=args.dpi)
    print(f"Saved sample to\n {fname}")
    plt.close()
    
    # plot sample as 2d histogram
    set_seaborn_style()
    cmap = "coolwarm"
    # cmap = sns.color_palette("Spectral", as_cmap=True)
    # plt.hist2d(
    #     sample[:, 0], sample[:, 1], bins=100, cmap=cmap
    # )
    sns.histplot(
        x=sample[:, 0], y=sample[:, 1], cmap=cmap, fill=True,
        bins=100
    )
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_hist2d.png"
    plt.savefig(fname, dpi=args.dpi)
    print(f"Saved hist2d to\n {fname}")
    plt.close()
    
    # KDE plot
    sns.kdeplot(
        x=sample[:, 0], y=sample[:, 1], cmap=cmap, fill=True
    )
    plt.tight_layout(pad=0.0)
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_kde.png"
    plt.savefig(fname, dpi=args.dpi)
    print(f"Saved kde to\n {fname}")
    plt.close()

def plot_density_traj_sidebyside(args: DictConfig) -> None:
    print("\n" + "-" * 80 + f"\nPlotting density sidebyside for {args.runname}\n")
    
    limmin = args.plim[0]
    limmax = args.plim[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    
    # folder with temporary trajectory plots
    tempdir = f"{args.savedir}/sidebyside"
    os.makedirs(tempdir, exist_ok=True)
    os.makedirs(PLOT_DIR_SOURCE, exist_ok=True)    
    
    tplotname = "logprob"
    combinedplotname = f"{tplotname}_{args['source']['trgt']}-to-{args.data['trgt']}"
    combinedplotname += f"-ot" if args.use_ot else ""
    
    nsamples = args.plot_batch_size
    n_t_span = args.plot_integration_steps
    n_img = 5 # number of images of the inference trajectory
    d_img = args.data.dims if "dims" in args.data else (1, 28, 28)
    
    
    model = get_model(**args.model).to(device)
    cp = torch.load(os.path.join(args.savedir, args.cpname + ".pth"), weights_only=True)
    model.load_state_dict(cp)
    
    trgtdist = get_dataset(**args.data)
    sourcedist = get_source_distribution(**args.source, trgtdist=trgtdist, device=device)

    # sample noise
    # sample = sample_8gaussians(nsamples) # [nsamples, 2]
    sample = sourcedist.sample(nsamples) # [nsamples, 2]
    plot_samples(args, sample)
    
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
    
    # integration times
    ts = torch.linspace(0, 1, n_img, device=device) # [n_img]
    
    sample = sourcedist.sample(nsamples) # [nsamples, 2]
    if args.classcond:
        # print(" -- Stopping sidebyside because class conditional with log-prob is not implemented yet")
        # return
        # compute trajectory once, later pick time slices for gif
        generated_class_list = torch.arange(10, device=device).repeat(nsamples // 10 + 1) # [100]
        generated_class_list = generated_class_list[:nsamples]
        y0 = sourcedist.sample(nsamples).to(device)
        # reshape to [nsamples, *d_img]
        y0 = y0.view([nsamples, *d_img])
        with torch.no_grad():
            traj = torchdiffeq.odeint(
                func=lambda t, x: model.forward(t, x, generated_class_list),
                y0=y0,
                # t is not the integration steps but the time points for which to solve for
                # The first element of is taken to be the initial time point
                t=ts.to(device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
                # for a fixed step solver, we need to specify the step size
                # method="euler", # euler, midpoint, rk4, heun3
                # options={"step_size": 1/intsteps},
            )
    # else:
    #     # nde = tdyn.NeuralODE(tdyn.DEFunc(torch_wrapper(model)), solver="dopri5").to(device)
    #     nde = tdyn.NeuralODE(tdyn.DEFunc(torch_wrapper(model)), solver="dopri5").to(device)
    #     # with torch.no_grad():
    #     # [n_img, nsamples, 2]
    #     traj = nde.trajectory(sample.to(device), t_span=ts.to(device))
    
    set_seaborn_style()
    # integrate up to t / starting form t using n_t_span steps
    for i, t in tqdm(enumerate(ts)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        if args.classcond:
            grid = make_grid(
                traj[i, :args.plot_nrows**2].view([-1, *d_img]).clip(-1, 1), 
                value_range=(-1, 1), padding=0, nrow=args.plot_nrows
            )
            img = ToPILImage()(grid)
            ax.imshow(img)
        else:
            ### log-probability / density plot
            nde = tdyn.NeuralODE(tdyn.DEFunc(CNF(model)), solver="euler", sensitivity="adjoint")
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
                    )[-1]
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
            
            # viridis coolwarm BuPu PuRd magma inferno cividis prism ocean
            cmap = "viridis" 
            # cmap = sns.color_palette("Spectral", as_cmap=True)
            ax.pcolormesh(
                X, Y, torch.exp(log_probs).cpu().numpy(), 
                # vmax=1, 
                cmap=cmap,
                # shading{'flat', 'nearest', 'gouraud', 'auto'}
                shading='gouraud', norm='linear'
            )
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(limmin, limmax)
        ax.set_ylim(limmin, limmax)
        # ax.set_title(f"{args.runname}", fontsize=30)
        # can't get rid of white border, so pad a bit to make it look cleaner
        plt.tight_layout(pad=0.1) 
        plt.savefig(f"{tempdir}/{tplotname}_{t:0.2f}.png", dpi=args.dpi)
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
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([]) 
        # torn off minor ticks
        ax.tick_params(axis='both', which='minor', length=0)
        # ax.set_title(f"T={ts[i]:0.2f}")
    
    plt.tight_layout(pad=0.0)
    fname = f"{args.savedir}/{combinedplotname}_sidebyside.png"
    plt.savefig(fname, dpi=args.dpi)
    plt.close()
    print(f"Saved sidebyside to\n {fname}")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_density_traj_sidebyside(args)


if __name__ == "__main__":
    hydra_wrapper()
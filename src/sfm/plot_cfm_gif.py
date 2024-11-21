import math
import os
import time
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

# import imageio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import Tensor
import torchdyn
from torchdyn.core import DEFunc, NeuralODE
from torchdyn.datasets import generate_moons
from torchdyn.nn import Augmenter

from torchcfm.models.models import MLP, GradModel
from torchcfm.utils import torch_wrapper

from sfm.distributions import get_source_distribution
from sfm.tcfmhelpers import CNF
from sfm.plotstyle import _cscheme

PLOT_DIR_SOURCE = "plots/sources"

def plot_cfm_gif(args: DictConfig) -> None:
    w = 7 # plot limits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    
    # folder with temporary trajectory plots
    os.makedirs(f"{args.savedir}/trajectory", exist_ok=True)
    
    tplotname = ""
    if args.doplot[0]:
        tplotname += "_logprob"
    if args.doplot[1]:
        tplotname += "_quiver"
    if args.doplot[2]:
        tplotname += "_traj"
    tplotname = tplotname[1:]
    
    gif_name = f"{tplotname}_{args['source']['trgt']}-to-{args.data['trgt']}"
    
    n_models = 1
    n_samples = 1024
    n_t_span = 201
    n_t_gif = 101
    
    # plotting stuff
    # log-prob plot
    points = 100j
    points_real = 100
    Y, X = np.mgrid[-w:w:points, -w:w:points] # [points, points]
    gridpoints = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1)).type(torch.float32) # [points**2, 2]
    # quiver plot
    points_small = 20j
    points_real_small = 20
    Y_small, X_small = np.mgrid[-w:w:points_small, -w:w:points_small] # [points_small, points_small]
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
    plt.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)
    plt.tight_layout()
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_scatter.png"
    plt.savefig(fname, dpi=40)
    print(f"Saved sample to\n {fname}")
    plt.close()
    
    # plot sample as 2d histogram
    plt.hist2d(sample[:, 0].numpy(), sample[:, 1].numpy(), bins=100, cmap="coolwarm")
    plt.tight_layout()
    fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_hist2d.png"
    plt.savefig(fname, dpi=40)
    print(f"Saved hist2d to\n {fname}")
    plt.close()
    
    ts = torch.linspace(0, 1, n_t_gif) # [n_t_gif]
    
    # compute trajectory once, later pick time slices for gif
    nde = NeuralODE(DEFunc(torch_wrapper(model)), solver="euler").to(device)
    # with torch.no_grad():
    # [n_t_gif, n_samples, 2]
    traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()
    
    n_plots = sum(args.doplot)
    for i, t in tqdm(enumerate(ts)):
        fig, axes = plt.subplots(n_plots, n_models, figsize=(6 * n_models, 6 * n_plots))
        axis = axes if n_models == 1 else axes[:, 0]
        iplot = 0
        
        ### log-probability / density plot
        # sample probability of points on a grid
        if args.doplot[0]:
            cnf = DEFunc(CNF(model))
            nde = NeuralODE(cnf, solver="euler", sensitivity="adjoint")
            cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
            with torch.no_grad():
                if t > 0:
                    # integrate backwards from t=1 (tdata) to t=0 (tnoise)
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
            log_probs = log_probs.reshape(Y.shape)
            ax = axis[iplot]
            ax.pcolormesh(X, Y, torch.exp(log_probs), vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-w, w)
            ax.set_ylim(-w, w)
            ax.set_title(f"{args.runname}", fontsize=30)
            iplot += 1
        
        ### Quiver plot
        if args.doplot[1]:
            # with torch.no_grad():
            out = model(
                torch.cat(
                    [gridpoints_small, torch.ones((gridpoints_small.shape[0], 1)) * t], dim=1
                ).to(device)
            )
            out = out.reshape([points_real_small, points_real_small, 2]).cpu().detach().numpy()
            ax = axis[iplot]
            ax.quiver(
                X_small,
                Y_small,
                out[:, :, 0],
                out[:, :, 1],
                np.sqrt(np.sum(out**2, axis=-1)),
                cmap="coolwarm",
                scale=50.0,
                width=0.015,
                pivot="mid",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-w, w)
            iplot += 1

        ### Trajectory plot
        if args.doplot[2]:
            ax = axis[iplot]
            ax.scatter(traj[:i, :, 0], traj[:i, :, 1], s=0.2, alpha=0.2, c=_cscheme["flow"])
            ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c=_cscheme["prior"])
            ax.scatter(traj[i, :, 0], traj[i, :, 1], s=4, alpha=1, c=_cscheme["final"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-w, w)
            ax.set_ylim(-w, w)
            iplot += 1
        
        plt.suptitle(f"{args['source']['trgt']} to Moons T={t:0.2f}", fontsize=40)
        plt.savefig(f"{args.savedir}/trajectory/{tplotname}_{t:0.2f}.png", dpi=40)
        plt.close()
        # print(f"Saved figure to\n {args.savedir}/trajectory/{t:0.2f}.png")
    
    # load all trajectory plots and save as gif
    fignames = [f"{args.savedir}/trajectory/{tplotname}_{t:0.2f}.png" for t in ts] 
    fignames += [f"{args.savedir}/trajectory/{tplotname}_{ts[-1].item():0.2f}.png"] * 10
    with imageio.get_writer(f"{args.savedir}/{gif_name}.gif", mode="I") as writer:
        for filename in fignames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Saved gif to\n {args.savedir}/{gif_name}.gif")


@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_cfm_gif(args)


if __name__ == "__main__":
    hydra_wrapper()
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
import torchdyn.core as tdyn
from torchdyn.nn import Augmenter

from torchcfm.models.models import MLP, GradModel
from torchcfm.utils import torch_wrapper

from sfm.distributions import get_source_distribution
from sfm.datasets import get_dataset
from sfm.tcfmhelpers import CNF
from sfm.plotstyle import _cscheme
from sfm.networks import get_model

import torchdiffeq
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

PLOT_DIR_SOURCE = "plots/sources"


def plot_cfm_gif(args: DictConfig) -> None:
    print("\n" + "-" * 80 + f"\nPlotting CFM gif for {args.runname}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.force_cpu:
        device = "cpu"
    
    # folder with temporary trajectory plots
    os.makedirs(f"{args.savedir}/gif", exist_ok=True)
    
    tplotname = ""
    if args.doplot[0]:
        tplotname += "_logprob"
    if args.doplot[1]:
        tplotname += "_quiver"
    if args.doplot[2]:
        tplotname += "_traj"
    tplotname = tplotname[1:]
    
    def get_frame_name(args: DictConfig, t: float):
        # f"{args.savedir}/gif/{tplotname}_{t:0.2f}.png"
        return f"{args.savedir}/gif/{tplotname}_{t:0.2f}.png"
    
    gif_name = f"{tplotname}_{args['source']['trgt']}-to-{args.data['trgt']}"
    gif_name += f"_is{args.plot_integration_steps}"
    gif_name += "_ot" if args.use_ot else ""
    
    n_models = 1 # number of columns in the plot
    nsamples = args.plot_batch_size
    n_t_span = args.plot_integration_steps
    n_frames = 101 # number of frames in the gif
    
    limmin = args.plim[0]
    limmax = args.plim[1]
    
    # plotting stuff
    # log-prob plot
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
    
    # model = MLP(dim=2, time_varying=True)
    model = get_model(**args.model).to(device)
    cp = torch.load(os.path.join(args.savedir, args.cpname + ".pth"), weights_only=True)
    model.load_state_dict(cp)
    
    trgtdist = get_dataset(**args.data)
    sourcedist = get_source_distribution(**args.source, trgtdist=trgtdist, device=device)

    # sample noise
    # sample = sample_8gaussians(nsamples) # [nsamples, D]
    sample = sourcedist.sample(nsamples) # [nsamples, D]
    sample = sample.cpu()
    
    os.makedirs(PLOT_DIR_SOURCE, exist_ok=True)
    
    # plot sample
    if args.source.data_dim == 2:
        plt.scatter(sample[:, 0], sample[:, 1], s=1, alpha=0.5)
        plt.tight_layout(pad=0.0)
        fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_scatter.png"
        plt.savefig(fname, dpi=args.dpi)
        print(f"Saved sample to\n {fname}")
        plt.close()
        
        # plot sample as 2d histogram
        plt.hist2d(sample[:, 0].numpy(), sample[:, 1].numpy(), bins=100, cmap="coolwarm")
        plt.tight_layout(pad=0.0)
        fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_hist2d.png"
        plt.savefig(fname, dpi=args.dpi)
        print(f"Saved hist2d to\n {fname}")
        plt.close()
    else:
        # plot sample as image
        d_img = args.data.dims
        grid = make_grid(
            sample[:args.plot_nrows**2].view([-1, *d_img]).clip(-1, 1), 
            value_range=(-1, 1), padding=0, nrow=args.plot_nrows
        )
        img = ToPILImage()(grid)
        plt.imshow(img)
        # remove ticks
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.0)
        fname = f"{PLOT_DIR_SOURCE}/{args['source']['trgt']}_img.png"
        plt.savefig(fname, dpi=args.dpi)
        print(f"Saved sample to\n {fname}")
        plt.close()
    
    
    sample = sourcedist.sample(nsamples) # [nsamples, D]
    ts = torch.linspace(0, 1, n_frames) # [n_frames]
    # compute trajectory once, later pick time slices for gif
    if args.classcond:
        # print(" -- Stopping gif early because class conditional is not implemented yet")
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
        grid = make_grid(
        # traj[-1, :100].view([-1, 1, 28, 28]
        traj[-1, :args.plot_nrows**2].view([-1, *d_img]).clip(-1, 1), 
            value_range=(-1, 1), padding=0, nrow=args.plot_nrows
        )
        img = ToPILImage()(grid)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.gca().axis("off")
        plt.tight_layout(pad=0.0)
        plt.savefig(f"{args.savedir}/gif/gentrajfinal.png", dpi=args.dpi)
        plt.close()
    else:
        nde = tdyn.NeuralODE(tdyn.DEFunc(torch_wrapper(model)), solver="euler").to(device)
        # with torch.no_grad():
        # [n_frames, nsamples, 2]
        traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()
    
    n_plots = sum(args.doplot) # number of rows in the plot
    for i, t in tqdm(enumerate(ts), total=n_frames, desc="Plotting gif frames"):
        fig, axes = plt.subplots(n_plots, n_models, figsize=(6 * n_models, 6 * n_plots))
        axis = axes if n_models == 1 else axes[:, 0]
        iplot = 0
        
        ### log-probability / density plot
        # sample probability of points on a grid
        if args.doplot[0]:
            if args.classcond:
                pass
            else:
                nde = tdyn.NeuralODE(tdyn.DEFunc(CNF(model)), solver="euler", sensitivity="adjoint")
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
                ax = axis[iplot] if n_plots > 1 else axis
                ax.pcolormesh(X, Y, torch.exp(log_probs).cpu().numpy(), vmax=1)
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(limmin, limmax)
                ax.set_ylim(limmin, limmax)
                # ax.set_title(f"{args.runname}", fontsize=20)
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
            ax = axis[iplot] if n_plots > 1 else axis
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
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(limmin, limmax)
            ax.set_ylim(limmin, limmax)
            iplot += 1

        ### Trajectory plot
        if args.doplot[2]:
            ax = axis[iplot] if n_plots > 1 else axis
            if args.classcond:
                grid = make_grid(
                    # traj[-1, :100].view([-1, 1, 28, 28]
                    traj[i, :args.plot_nrows**2].view([-1, *d_img]).clip(-1, 1), 
                    value_range=(-1, 1), padding=0, nrow=args.plot_nrows
                )
                img = ToPILImage()(grid)
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.scatter(traj[:i, :, 0], traj[:i, :, 1], s=0.2, alpha=0.2, c=_cscheme["flow"])
                ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c=_cscheme["prior"])
                ax.scatter(traj[i, :, 0], traj[i, :, 1], s=4, alpha=1, c=_cscheme["final"])
                ax.axis("off")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(limmin, limmax)
                ax.set_ylim(limmin, limmax)
            iplot += 1
        
        plt.tight_layout(pad=0.0)
        # plt.suptitle(f"{args['source']['trgt']} to Moons T={t:0.2f}", fontsize=20)
        plt.savefig(get_frame_name(args, t), dpi=args.dpi)
        plt.close()
        # print(f"Saved figure to\n {args.savedir}/gif/{t:0.2f}.png")
    
    # load all trajectory plots and save as gif
    fignames = [get_frame_name(args, t) for t in ts] 
    # add 10 frames at the end to make the gif loop
    fignames += [get_frame_name(args, ts[-1].item())] * 10
    
    # subrectangles to reduce file size
    with imageio.get_writer(f"{args.savedir}/{gif_name}.gif", mode="I", subrectangles=True) as writer:
        for filename in fignames:
            image = imageio.imread(filename)
            # Crop any border/shadow by finding the non-white pixels
            # nonwhite = np.where(image[:,:,0] < 255)
            # if len(nonwhite[0]) > 0:
            #     ymin, ymax = np.min(nonwhite[0]), np.max(nonwhite[0])
            #     xmin, xmax = np.min(nonwhite[1]), np.max(nonwhite[1])
            #     image = image[ymin:ymax+1, xmin:xmax+1]
            writer.append_data(image)
    print(f"Saved gif to\n {args.savedir}/{gif_name}.gif")


@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_cfm_gif(args)


if __name__ == "__main__":
    hydra_wrapper()
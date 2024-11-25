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
import torchdiffeq
import torchdyn
import torchdyn.core as tdyn
from torchdyn.nn import Augmenter
from omegaconf import DictConfig
from tqdm import tqdm

from torchcfm.utils import torch_wrapper

from sfm.distributions import get_source_distribution
from sfm.tcfmhelpers import CNF, plot_trajectories
from sfm.plotstyle import _cscheme, set_seaborn_style
from sfm.networks import get_model
from sfm.datasets import get_dataset

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

# for node.trajcetory
# SOLVER_DICT = {
#     'euler': Euler, 'midpoint': Midpoint,
#     'rk4': RungeKutta4, 'rk-4': RungeKutta4, 'RungeKutta4': RungeKutta4,
#     'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
#     'tsit5': Tsitouras45, 'Tsitouras45': Tsitouras45, 'Tsitouras5': Tsitouras45,
#     'ieuler': ImplicitEuler, 'implicit_euler': ImplicitEuler,
#     'alf': AsynchronousLeapfrog, 'AsynchronousLeapfrog': AsynchronousLeapfrog
# }

def plot_integration_steps(args: DictConfig) -> None:
    print("\n" + "-"*80 + f"\nPlotting integration steps for {args.runname}\n")
    
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
    nsamples = args.plot_batch_size
    max_integration_steps = args.plot_integration_steps
    n_integration_steps = 5
    n_img = 5 # number of images of the inference trajectory
    tdata = 1
    tnoise = 0
    
    model = get_model(**args.model).to(device)
    cp = torch.load(os.path.join(args.savedir, args.cpname + ".pth"), weights_only=True)
    model.load_state_dict(cp)
    
    trgtdist = get_dataset(**args.data, device=device)
    sourcedist = get_source_distribution(**args.source, trgtdist=trgtdist, device=device)

    # sample noise
    # sample = sample_8gaussians(nsamples) # [nsamples, 2]
    sample = sourcedist.sample(nsamples) # [nsamples, 2]
    
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
    
    # Try multiple adaptive solvers and average NFE over multiple runs
    solvers = ["dopri5"]  # rk4 is another adaptive solver
    n_runs = 5
    nfe_results = {solver: [] for solver in solvers}
    nsolsteps_results = {solver: [] for solver in solvers}

    for solver in solvers:
        print(f"\nTesting {solver} solver:")
        for run in range(n_runs):
            if args.classcond:
                generated_class_list = torch.arange(10, device=device).repeat(nsamples // 10 + 1) 
                generated_class_list = generated_class_list[:nsamples]
                y0 = sourcedist.sample(nsamples).to(device)
                # reshape to [nsamples, 1, 28, 28]
                y0 = y0.view(nsamples, 1, 28, 28)
                model.nfe = 0
                with torch.no_grad():
                    # Fixed solvers (euler, midpoint, rk4, explicit_adams, implicit_adams)
                    # Adaptive solvers (dopri8, dopri5, bosh3, adaptive_heun
                    traj = torchdiffeq.odeint(
                        func=lambda t, x: model.forward(t, x, generated_class_list),
                        y0=y0,
                        t=torch.linspace(0, 1, 2, device=device),
                        atol=1e-4,
                        rtol=1e-4,
                        method=solver,
                    )
                nfe_results[solver].append(model.nfe)
                print(f"Run {run+1}: NFE={model.nfe}")
            else:
                ts = torch.linspace(0, 1, 2)
                model.nfe = 0
                # SOLVER_DICT = {'euler': Euler, 'midpoint': Midpoint,
                #    'rk4': RungeKutta4, 'rk-4': RungeKutta4, 'RungeKutta4': RungeKutta4,
                #    'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
                #    'tsit5': Tsitouras45, 'Tsitouras45': Tsitouras45, 'Tsitouras5': Tsitouras45,
                #    'ieuler': ImplicitEuler, 'implicit_euler': ImplicitEuler,
                #    'alf': AsynchronousLeapfrog, 'AsynchronousLeapfrog': AsynchronousLeapfrog}
                nde = tdyn.NeuralODE(tdyn.DEFunc(torch_wrapper(model)), solver=solver).to(device)
                x, t_span = nde._prep_integration(sample.to(device), ts.to(device))
                t_eval, traj = torchdyn.numerics.odeint(nde.vf, x, t_span, solver=nde.solver, atol=nde.atol, rtol=nde.rtol, return_all_eval=True)
                traj = traj.detach().cpu().numpy()
                nsolsteps = len(t_eval)
                nfe_results[solver].append(model.nfe)
                nsolsteps_results[solver].append(nsolsteps)
                print(f"Run {run+1}: NFE={model.nfe}, NSolSteps={nsolsteps}")

    # Calculate and print averages
    for solver in solvers:
        avg_nfe = sum(nfe_results[solver]) / len(nfe_results[solver])
        std_nfe = np.std(nfe_results[solver])
        print(f"\n{solver} average NFE: {avg_nfe:.1f} ± {std_nfe:.1f}")
        if not args.classcond:
            avg_nsolsteps = sum(nsolsteps_results[solver]) / len(nsolsteps_results[solver])
            std_nsolsteps = np.std(nsolsteps_results[solver])
            print(f"{solver} average NSolSteps: {avg_nsolsteps:.1f} ± {std_nsolsteps:.1f}")

    # Save results
    results = {
        'nfe': nfe_results,
        'nsolsteps': nsolsteps_results if not args.classcond else None
    }
    np.save(f"{tempdir}/solver_results.npy", results)
    
    # integrate up to 1 / starting from 1 using intsteps steps
    # the better the flow the faster / fewer steps still give good results
    # to fix the number of steps we need to use a fixed step solver
    # intsteps will include 0 and max
    intsteps_list = torch.linspace(0, max_integration_steps, n_integration_steps)
    intsteps_list = [int(intsteps) for intsteps in intsteps_list]
    intsteps_list = [intsteps if intsteps != 1 else 2 for intsteps in intsteps_list]
    print(f"intsteps_list: {intsteps_list}")
    logprobs_intsteps = []
    for nsteps in tqdm(intsteps_list): 
        
        ### calculate log-likelihood of sample
        if args.classcond:
            pass
        else:
            cnf = tdyn.DEFunc(CNF(model))
            nde = tdyn.NeuralODE(cnf, solver="euler", sensitivity="adjoint")
            cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
            # with torch.no_grad():
            # x1 = sample_moons(nsamples).to(device).requires_grad_()
            x1 = trgtdist.sample(nsamples).to(device).requires_grad_()
            if nsteps > 0:
                # if intsteps == 1:
                #     print("Warning: setting intsteps=2 because intsteps=1 is not allowed")
                #     intsteps = 2
                # integrate backwards from t=1 (tdata) to t=0 (tnoise)
                aug_traj = (
                cnf_model[1]
                .to(device)
                .trajectory(
                    x=Augmenter(1, 1)(x1).to(device),
                    t_span=torch.linspace(start=tdata, end=tnoise, steps=nsteps, device=device),
                    )
                )[-1]
                # Compute log probabilities
                # We can load the log probs later to plot the training progress
                log_probs = sourcedist.log_prob(aug_traj[:, 1:]) - aug_traj[:, 0]
            else:
                log_probs = sourcedist.log_prob(x1)
            logprobs_intsteps.append([nsteps, log_probs.nanmean().item()])
            print(f"Log-likelihood intsteps={nsteps}: {log_probs.nanmean().item():0.3f}")

        ### trajectory / generated sample
        # src/sfm/plot_cfm_gif.py
        ts = torch.linspace(0, 1, nsteps) # [intsteps]
        if args.classcond:
            d_img = (1, 28, 28)
            if nsteps > 0:
                generated_class_list = torch.arange(10, device=device).repeat(nsamples // 10 + 1) # [100]
                generated_class_list = generated_class_list[:nsamples]
                y0 = sourcedist.sample(nsamples).to(device)
                # reshape to [nsamples, *d_img]
                y0 = y0.view([nsamples, *d_img])
                model.nfe = 0
                with torch.no_grad():
                    # [nsteps, nsamples, *d_img]
                    traj = torchdiffeq.odeint(
                        func=lambda t, x: model.forward(t, x, generated_class_list),
                        y0=y0,
                        # t is not the integration steps but the time points for which to solve for
                        # The first element of is taken to be the initial time point
                        t=ts.to(device),
                        # atol=1e-4,
                        # rtol=1e-4,
                        method="euler",
                        # for a fixed step solver, we need to specify the step size
                        # method="euler", # euler, midpoint, rk4, heun3
                        options={"step_size": 1/nsteps},
                    )[-1]
                    # reshape to [nsamples, h*w]
                    traj = traj.view(nsamples, -1)
                print(f"Euler integration: nsteps={nsteps} NFE={model.nfe}")
                assert model.nfe <= nsteps + 1, f"NFE={model.nfe} > nsteps={nsteps} + 1"
            else:
                # no integration, just evaluate at t=0
                # [nsamples, h*w]
                traj = sourcedist.sample(nsamples)
            grid = make_grid(
                traj[:100].view([-1, *d_img]).clip(-1, 1), 
                value_range=(-1, 1), padding=0, nrow=10
            )
            img = ToPILImage()(grid)
            plt.imshow(img)
            plt.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout(pad=0.0)
        else:
            if nsteps > 0:
                model.nfe = 0
                nde = tdyn.NeuralODE(tdyn.DEFunc(torch_wrapper(model)), solver="euler").to(device)
                # [intsteps, nsamples, D]
                traj = nde.trajectory(sample.to(device), t_span=ts.to(device)).detach().cpu().numpy()
                print(f"Euler integration: nsteps={nsteps} NFE={model.nfe}")
                assert model.nfe <= nsteps + 1, f"NFE={model.nfe} > nsteps={nsteps} + 1"
            else:
                traj = sourcedist.sample(nsamples)
            fig = plot_trajectories(traj)
            # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            # # i is the frame number
            # ax.scatter(traj[:i, :, 0], traj[:i, :, 1], s=0.2, alpha=0.2, c=_cscheme["flow"])
            # ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.8, c=_cscheme["prior"])
            # ax.scatter(traj[i, :, 0], traj[i, :, 1], s=4, alpha=1, c=_cscheme["final"])
            # ax.axis("off")
            # ax.set_aspect('equal') # set quadratic aspect ratio
            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.set_xlim(limmin, limmax)
            # ax.set_ylim(limmin, limmax)
        plt.tight_layout(pad=0.0)
        plt.savefig(f"{tempdir}/gen_intsteps{nsteps}.png")
        # print(f"Saved gen to {tempdir}/gen_intsteps{intsteps}.png")
        plt.close()
        
        ### log-probability / density plot
        if args.classcond:
            pass
        else:
            fig, axis = plt.subplots(1, 1, figsize=(6, 6))
            cnf = tdyn.DEFunc(CNF(model))
            # adjoint sensitivity is needed for the backward pass # sensitivity="adjoint"
            nde = tdyn.NeuralODE(cnf, solver="euler")
            cnf_model = torch.nn.Sequential(Augmenter(augment_idx=1, augment_dims=1), nde)
            with torch.no_grad():
                if nsteps > 0:
                    # integrate backwards from t to 0
                    aug_traj = (
                        cnf_model[1]
                        .to(device)
                        .trajectory(
                            Augmenter(1, 1)(gridpoints).to(device),
                            t_span=torch.linspace(start=1, end=0, steps=nsteps, device=device),
                        )
                    )[-1]
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
                X, Y, torch.exp(log_probs).cpu().numpy(), 
                # vmax=1, 
                cmap=cmap,
                # shading{'flat', 'nearest', 'gouraud', 'auto'}
                shading='gouraud', norm='linear'
            )
            ax.axis("off")
            ax.set_aspect('equal') # set quadratic aspect ratio
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(limmin, limmax)
            ax.set_ylim(limmin, limmax)
            # ax.set_title(f"{args.runname}", fontsize=30)
            # can't get rid of white border, so pad a bit to make it look cleaner
            plt.tight_layout(pad=0.0) 
            plt.savefig(f"{tempdir}/{tplotname}_{nsteps}.png", dpi=args.dpi)
            plt.close()
    
    ### log-probability / density plot side by side
    if not args.classcond:
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
            ax.axis("off")
            ax.set_aspect('equal') # set quadratic aspect ratio
            ax.set_xticks([])
            ax.set_yticks([]) 
            # torn off minor ticks
            ax.tick_params(axis='both', which='minor', length=0)
            # ax.set_title(f"T={ts[i]:0.2f}")
        
        plt.tight_layout(pad=0.0)
        fname = f"{args.savedir}/intsteps_density.png"
        plt.savefig(fname, dpi=args.dpi)
        plt.close()
        print(f"Saved integration steps to\n {fname}")
    
    ### trajectory / generated sample side by side
    fignames = [f"{tempdir}/gen_intsteps{intsteps}.png" for intsteps in intsteps_list]
    images = [imageio.imread(fname) for fname in fignames]
    
    # create a new figure with subplots side by side
    n_plots = len(images)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    # plot each image
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        # remove border between plots
        ax.axis("off")
        ax.set_aspect('equal') # set quadratic aspect ratio
        ax.set_xticks([])
        ax.set_yticks([]) 
        # torn off minor ticks
        ax.tick_params(axis='both', which='minor', length=0)
        # ax.set_title(f"T={ts[i]:0.2f}")
    plt.tight_layout(pad=0.0)
    fname = f"{args.savedir}/intsteps_gen.png"
    plt.savefig(fname, dpi=args.dpi)
    plt.close()
    print(f"Saved generation to\n {fname}")
    
    # save intsteps_list for reference
    np.save(f"{args.savedir}/intsteps_list.npy", intsteps_list)
    # print(f"Saved intsteps_list to {args.savedir}/{combinedplotname}_intsteps_list.npy")
    
    if not args.classcond:
        # save logprobs_intsteps
        logprobs_intsteps = np.array(logprobs_intsteps)
        np.save(f"{args.savedir}/logprobs_intsteps.npy", logprobs_intsteps)
        # print(f"Saved logprobs_intsteps to {args.savedir}/{combinedplotname}_logprobs_intsteps.npy")


@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    plot_integration_steps(args)


if __name__ == "__main__":
    hydra_wrapper()
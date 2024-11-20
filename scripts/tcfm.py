# a common file to call train_cfm.py and plot_cfm_gif.py
# parses args, sets names, etc.

import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from sfm.train_cfm import train_cfm
from sfm.plot_cfm_gif import plot_cfm_gif

@hydra.main(config_name="tcfm", config_path="../src/sfm/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    
    # set up names for files and directories
    proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    args.runname = f"{args['source']['trgt']}-{args.data['trgt']}"
    if args.use_ot:
        args.runname += "-ot"
    args.savedir = f"{proj_dir}/runs/{args.runname}"
    args.cpname = f"{args.cpname}_{args.n_trainsteps}"
    os.makedirs(args.savedir, exist_ok=True)
    
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # run the task
    if args.task == "train":
        train_cfm(args)
    elif args.task == "gif":
        plot_cfm_gif(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    hydra_wrapper()

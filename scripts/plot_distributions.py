import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
import os
import omegaconf

# Import your distributions
from sfm.distributions import get_source_distribution

if __name__ == "__main__":
    # load distributions from yaml files 
    currentfile = os.path.abspath(__file__)
    currentdir = os.path.dirname(currentfile)
    source_dir = os.path.abspath(f"{currentdir}/../src/sfm/config/source/")

    distributions = {}
    for fname in os.listdir(source_dir):
        with open(f"{source_dir}/{fname}", "r") as f:
            cfg = omegaconf.OmegaConf.load(f)
            name = os.path.splitext(fname)[0].replace(".yaml", "")
            distributions[name] = cfg["source"]

    n_samples = 1000

    # Plot and save each distribution
    for name, params in distributions.items():
        print(f"Plotting {name.capitalize()} Distribution")
        plt.title(f"{name.capitalize()} Distribution")
        fname = f"plots/sources/{name}_distribution.png"

        dist = get_source_distribution(**params)
        samples = dist.sample(n_samples)
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        assert samples.shape == (n_samples, 2), f"Samples have wrong shape {samples.shape}"

        # plt.scatter(samples[:, 0], samples[:, 1])
        # plt.show()

        plt.figure(figsize=(4.8, 4.8), dpi=150)
        plt.hist2d(*samples.T, bins=64)
        plt.savefig(fname)
        print(f" saved {fname}")
        plt.close()

    # # Generate yaml files for each distribution
    # currentfile = os.path.abspath(__file__)
    # for name, params in distributions.items():
    #     fname = f"{os.path.dirname(currentfile)}/../src/sfm/config/source/{name}.yaml"

    #     # Convert torch tensors to lists for YAML serialization
    #     config = {}
    #     for k, v in params.items():
    #         if hasattr(v, 'tolist'):
    #             config[k] = v.tolist()
    #         else:
    #             config[k] = v

    #     # Write the yaml file
    #     with open(fname, 'w') as f:
    #         # @package _global_
    #         # add this line to the top of the file
    #         f.write("# @package _global_\n\n")
    #         f.write("source: \n  ")
    #         # indent the rest of the file
    #         f.write(
    #             yaml.dump(
    #                 config, default_flow_style=False
    #             ).replace("\n", "\n  ")
    #         )
    #     print(f"Created {fname}")

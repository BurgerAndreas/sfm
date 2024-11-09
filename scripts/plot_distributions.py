import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml
import os

# Import your distributions
from sfm.distributions import get_source_distribution

# Define the distributions and their parameters
distributions = {
    "normal": {"type": "normal"},
    "gaussian": {"type": "gaussian", "mu": torch.tensor([0.0, 0.0]), "Sigma": torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
    "isotropic": {"type": "isotropic", "mu": torch.tensor([2.0, 1.0]), "sigma": 0.5},
    "diagonal": {"type": "diagonal", "mu": torch.tensor([0.0, 0.0]), "sigma": torch.tensor([1.0, 0.5])},
    "beta": {"type": "beta", "alpha": 2.0, "beta": 2.0},
    "mog": {
        "type": "mog",
        "mus": torch.tensor([[0.0, 0.0], [5.0, 5.0]]),
        "sigmas": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        "pis": torch.tensor([0.5, 0.5]),
    },
    "cauchy": {"type": "cauchy", "loc": 0.0, "scale": 1.0},
    "chi2": {"type": "chi2", "df": 5.0},
    "dirichlet": {"type": "dirichlet", "concentration": torch.tensor([1.0, 2.0])},
    "exponential": {"type": "exponential", "rate": 0.5},
    "fisher": {"type": "fisher", "df1": 5.0, "df2": 10.0},
    "gamma": {"type": "gamma", "concentration": 2.0, "rate": 2.0},
    "gumbel": {"type": "gumbel", "loc": 0.0, "scale": 1.0},
    "halfcauchy": {"type": "halfcauchy", "scale": 1.0},
    "halfnormal": {"type": "halfnormal", "scale": 1.0},
    "inversegamma": {"type": "inversegamma", "concentration": 2.0, "rate": 2.0},
    "kumaraswamy": {"type": "kumaraswamy", "a": 2.0, "b": 2.0},
    "laplace": {"type": "laplace", "loc": 0.0, "scale": 1.0},
    "lognormal": {"type": "lognormal", "loc": 0.0, "scale": 1.0},
    "lowrank": {
        "type": "lowrank",
        "loc": torch.tensor([0.0, 0.0]),
        "cov_factor": torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
        "cov_diag": torch.tensor([1.0, 1.0]),
    },
    "multivariate": {
        "type": "multivariate",
        "loc": torch.tensor([0.0, 0.0]),
        "covariance_matrix": torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
    },
    "pareto": {"type": "pareto", "scale": 1.0, "alpha": 2.0},
    "relaxedbernoulli": {"type": "relaxedbernoulli", "logits": torch.tensor([0.0, 2.0]), "temperature": 1.0},
    "studentt": {"type": "studentt", "df": 5.0, "loc": 0.0, "scale": 1.0},
    "vonmises": {"type": "vonmises", "loc": 0.0, "concentration": 1.0},
    "weibull": {"type": "weibull", "scale": 1.0, "concentration": 2.0},
}


if __name__ == "__main__":
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
    

    # Generate yaml files for each distribution
    currentfile = os.path.abspath(__file__)
    for name, params in distributions.items():
        fname = f"{os.path.dirname(currentfile)}/../src/sfm/config/source/{name}.yaml"
        
        # Convert torch tensors to lists for YAML serialization
        config = {}
        for k, v in params.items():
            if hasattr(v, 'tolist'):
                config[k] = v.tolist()
            else:
                config[k] = v
        
        # Write the yaml file
        with open(fname, 'w') as f:
            # @package _global_
            # add this line to the top of the file
            f.write("# @package _global_\n\n")
            f.write("source: \n  ")
            # indent the rest of the file   
            f.write(
                yaml.dump(
                    config, default_flow_style=False
                ).replace("\n", "\n  ")
            )
        print(f"Created {fname}")

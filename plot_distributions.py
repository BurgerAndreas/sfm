import matplotlib.pyplot as plt
import torch
import numpy as np

# Import your distributions
from distributions import get_source_distribution

# Define the distributions and their parameters
distributions = {
    "normal": {"type": "normal"},
    "gaussian": {"type": "gaussian", "mu": torch.tensor([0.0, 0.0]), "Sigma": torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
    "isotropic": {"type": "isotropic", "mu": 2.0, "sigma": 0.5},
    "diagonal": {"type": "diagonal", "mu": torch.tensor([0.0, 0.0]), "sigma": torch.tensor([1.0, 0.5])},
    "beta": {"type": "beta", "alpha": 2.0, "beta": 2.0},
    "mog": {"type": "mog", "mus": [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 5.0])], "sigmas": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])], "pis": [0.5, 0.5]},
    "cauchy": {"type": "cauchy", "loc": 0.0, "scale": 1.0},
    "chi2": {"type": "chi2", "df": 5.0},
    "dirichlet": {"type": "dirichlet", "concentration": torch.tensor([1.0, 2.0, 3.0])},
    "exponential": {"type": "exponential", "rate": 0.5},
    "fisher": {"type": "fisher", "df1": 5.0, "df2": 10.0},
    "gamma": {"type": "gamma", "concentration": 2.0, "rate": 2.0},
    "gumbel": {"type": "gumbel", "loc": 0.0, "scale": 1.0},
    "halfcauchy": {"type": "halfcauchy", "scale": 1.0},
    "halfnormal": {"type": "halfnormal", "scale": 1.0},
    "inversegamma": {"type": "inversegamma", "concentration": 2.0, "rate": 2.0},
    "kumaraswamy": {"type": "kumaraswamy", "a": 2.0, "b": 2.0},
    "lkj": {"type": "lkj", "dim": 3, "concentration": 1.0},
    "laplace": {"type": "laplace", "loc": 0.0, "scale": 1.0},
    "lognormal": {"type": "lognormal", "loc": 0.0, "scale": 1.0},
    "lowrank": {"type": "lowrank", "loc": torch.tensor([0.0, 0.0]), "cov_factor": torch.tensor([[1.0, 0.5], [0.5, 1.0]]), "cov_diag": torch.tensor([1.0, 1.0])},
    "multivariate": {"type": "multivariate", "loc": torch.tensor([0.0, 0.0]), "covariance_matrix": torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
    "pareto": {"type": "pareto", "scale": 1.0, "alpha": 2.0},
    "relaxedbernoulli": {"type": "relaxedbernoulli", "logits": torch.tensor([0.0]), "temperature": 1.0},
    "studentt": {"type": "studentt", "df": 5.0, "loc": 0.0, "scale": 1.0},
    "vonmises": {"type": "vonmises", "loc": 0.0, "concentration": 1.0},
    "weibull": {"type": "weibull", "scale": 1.0, "concentration": 2.0},
    "wishart": {"type": "wishart", "df": 5.0, "scale": torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
}

# distributions = {
#     'normal': [
#         {'loc': 0.0, 'scale': 1.0},
#         {'loc': 2.0, 'scale': 0.5},
#         {'loc': -1.0, 'scale': 2.0}
#     ],
#     'gaussian': [
#         {'mu': torch.tensor([0.0, 0.0]), 'Sigma': torch.tensor([[1.0, 0.5], [0.5, 1.0]])},
#         {'mu': torch.tensor([2.0, 2.0]), 'Sigma': torch.tensor([[2.0, 0.0], [0.0, 2.0]])},
#         {'mu': torch.tensor([-1.0, -1.0]), 'Sigma': torch.tensor([[0.5, 0.0], [0.0, 0.5]])}
#     ],
#     'beta': [
#         {'concentration1': 2.0, 'concentration0': 2.0},
#         {'concentration1': 1.0, 'concentration0': 3.0},
#         {'concentration1': 3.0, 'concentration0': 1.0}
#     ],
#     'cauchy': [
#         {'loc': 0.0, 'scale': 1.0},
#         {'loc': 2.0, 'scale': 0.5},
#         {'loc': -1.0, 'scale': 2.0}
#     ],
#     'chi2': [
#         {'df': 5.0},
#         {'df': 10.0},
#         {'df': 2.0}
#     ],
#     'dirichlet': [
#         {'concentration': torch.tensor([1.0, 2.0, 3.0])},
#         {'concentration': torch.tensor([2.0, 1.0, 3.0])},
#         {'concentration': torch.tensor([3.0, 2.0, 1.0])}
#     ],
#     'exponential': [
#         {'rate': 0.5},
#         {'rate': 1.0},
#         {'rate': 2.0}
#     ],
#     'gamma': [
#         {'concentration': 2.0, 'rate': 2.0},
#         {'concentration': 1.0, 'rate': 3.0},
#         {'concentration': 3.0, 'rate': 1.0}
#     ],
#     'gumbel': [
#         {'loc': 0.0, 'scale': 1.0},
#         {'loc': 2.0, 'scale': 0.5},
#         {'loc': -1.0, 'scale': 2.0}
#     ],
#     'halfcauchy': [
#         {'scale': 1.0},
#         {'scale': 0.5},
#         {'scale': 2.0}
#     ],
#     'halfnormal': [
#         {'scale': 1.0},
#         {'scale': 0.5},
#         {'scale': 2.0}
#     ],
#     'inversegamma': [
#         {'concentration': 2.0, 'rate': 2.0},
#         {'concentration': 1.0, 'rate': 3.0},
#         {'concentration': 3.0, 'rate': 1.0}
#     ],
#     'kumaraswamy': [
#         {'concentration1': 2.0, 'concentration0': 2.0},
#         {'concentration1': 1.0, 'concentration0': 3.0},
#         {'concentration1': 3.0, 'concentration0': 1.0}
#     ],
#     'laplace': [
#         {'loc': 0.0, 'scale': 1.0},
#         {'loc': 2.0, 'scale': 0.5},
#         {'loc': -1.0, 'scale': 2.0}
#     ],
#     'lognormal': [
#         {'loc': 0.0, 'scale': 1.0},
#         {'loc': 2.0, 'scale': 0.5},
#         {'loc': -1.0, 'scale': 2.0}
#     ],
#     'pareto': [
#         {'scale': 1.0, 'alpha': 2.0},
#         {'scale': 0.5, 'alpha': 3.0},
#         {'scale': 2.0, 'alpha': 1.0}
#     ],
#     'studentt': [
#         {'df': 5.0, 'loc': 0.0, 'scale': 1.0},
#         {'df': 10.0, 'loc': 2.0, 'scale': 0.5},
#         {'df': 2.0, 'loc': -1.0, 'scale': 2.0}
#     ],
#     'vonmises': [
#         {'loc': 0.0, 'concentration': 1.0},
#         {'loc': 2.0, 'concentration': 0.5},
#         {'loc': -1.0, 'concentration': 2.0}
#     ],
#     'weibull': [
#         {'scale': 1.0, 'concentration': 2.0},
#         {'scale': 0.5, 'concentration': 3.0},
#         {'scale': 2.0, 'concentration': 1.0}
#     ],
# }d

# Plot and save each distribution
for name, params in distributions.items():
    print(f"Plotting {name.capitalize()} Distribution")
    dist = get_source_distribution(**params)
    x = torch.linspace(-5, 5, 1000)
    y = torch.exp(dist.log_prob(x))
    plt.figure()
    plt.plot(x, y)
    plt.title(f"{name.capitalize()} Distribution")
    plt.savefig(f"{name}_distribution.png")

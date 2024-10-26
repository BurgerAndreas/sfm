import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from torch import Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import odeint

# To change the source distribution, change
# 1. the sampling during training and testing
# 2. the log-probability density function

# Log probability density function for a normal distribution
# Uses the definition from Appendix C in the paper, where the 
# CNF models the probability density path and reshapes the prior density
# Eq. (27) in the paper (for the push-forward equation)

class SourceDistribution:
    def __init__(self, **kwargs):
        pass

    def log_prob(self, x: Tensor) -> Tensor:
        return None
    
    def sample(self, shape: int | tuple) -> Tensor:
        raise NotImplementedError
    
def log_normal(x: Tensor) -> Tensor:
    """multivariate standard normal distribution, where the mean is 00 and the variance (or standard deviation squared) is 11 along every dimension."""
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

def log_isotropic_normal(x: Tensor, mu: float, sigma: float) -> Tensor:
    # normalization constant klog⁡(2πσ^2), where x.size(-1) gives the dimensionality kk
    normalization_term = x.size(-1) * math.log(2 * math.pi * sigma**2)
    squared_term = ((x - mu).square().sum(dim=-1)) / sigma**2
    return -(squared_term + normalization_term) / 2

def log_diagonal_normal(x: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    # assumes that the covariance matrix is diagonal. If the covariance matrix has off-diagonal terms (implying correlations between dimensions), 
    # you'd need to use the full multivariate Gaussian formula with the inverse of the covariance matrix and its determinant
    normalization_term = torch.log(2 * math.pi * sigma**2).sum(dim=-1)
    squared_term = (((x - mu).square()) / sigma**2).sum(dim=-1)
    return -(squared_term + normalization_term) / 2

def log_normal(x: Tensor, mu: Tensor, Sigma: Tensor) -> Tensor:
    # Compute the Mahalanobis distance
    diff = x - mu
    inv_Sigma = torch.inverse(Sigma)
    mahalanobis_distance = (diff.unsqueeze(-1).transpose(-1, -2) @ inv_Sigma @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    # Compute the log determinant of the covariance matrix
    log_det_Sigma = torch.logdet(Sigma)
    # Number of dimensions
    k = x.size(-1)
    # Log probability calculation
    return -0.5 * (mahalanobis_distance + log_det_Sigma + k * math.log(2 * math.pi))

def log_beta(x: Tensor, alpha: float, beta: float) -> Tensor:
    beta_func = torch.lgamma(torch.tensor(alpha)) + torch.lgamma(torch.tensor(beta)) - torch.lgamma(torch.tensor(alpha + beta))
    return ((alpha - 1) * x.log() + (beta - 1) * (1 - x).log() - beta_func).sum(dim=-1)

class StandardNormalSource(SourceDistribution):
    def __init__(self, **kwargs):
        """Mean=0, std=1"""
        pass

    def log_prob(self, x: Tensor) -> Tensor:
        return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

    def sample(self, shape: int | tuple) -> Tensor:
        return torch.randn(shape)

class IsotropicGaussianSource(SourceDistribution):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, **kwargs):
        """Isotropic Gaussian with mean=mu and std=sigma"""
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x: Tensor) -> Tensor:
        # Gaussian log probability for isotropic Gaussian with mean=mu and std=sigma
        normalization_term = math.log(2 * math.pi * self.sigma**2)
        squared_term = ((x - self.mu).square()).sum(dim=-1) / self.sigma**2
        return -(squared_term + normalization_term * x.size(-1)) / 2

    def sample(self, shape: int | tuple) -> Tensor:
        # Sample from Gaussian with mean=mu and std=sigma
        return self.mu + self.sigma * torch.randn(shape)

class DiagonalGaussianSource(SourceDistribution):
    def __init__(self, mu: Tensor, sigma: Tensor, **kwargs):
        """
        mu: Tensor of shape (d,) where d is the dimensionality of the distribution
        sigma: Tensor of shape (d,) representing the standard deviation for each dimension
        """
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Computes the log probability for a diagonal Gaussian distribution.
        x: Tensor of shape (n, d) where n is the batch size and d is the dimensionality.
        """
        normalization_term = torch.log(2 * math.pi * self.sigma**2).sum(dim=-1)
        squared_term = (((x - self.mu) ** 2) / (self.sigma ** 2)).sum(dim=-1)
        return -(squared_term + normalization_term) / 2

    def sample(self, shape: int | tuple) -> Tensor:
        """
        Samples from a diagonal Gaussian distribution.
        shape: Tuple representing the shape of the samples (batch_size, d).
        """
        return self.mu + self.sigma * torch.randn(shape)

class GaussianSource(SourceDistribution):
    def __init__(self, mu: Tensor, Sigma: Tensor, **kwargs):
        """
        mu: Tensor of shape (d,) where d is the dimensionality of the distribution.
        Sigma: Tensor of shape (d, d), the covariance matrix.
        """
        self.mu = mu
        self.Sigma = Sigma
        self.inv_Sigma = torch.inverse(Sigma)  # Precompute the inverse of the covariance matrix
        self.log_det_Sigma = torch.logdet(Sigma)  # Precompute the log determinant

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Computes the log probability for a general multivariate Gaussian distribution.
        x: Tensor of shape (n, d) where n is the batch size and d is the dimensionality.
        """
        diff = x - self.mu
        mahalanobis_term = (diff.unsqueeze(-1).transpose(-1, -2) @ self.inv_Sigma @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        
        # Compute the number of dimensions
        k = x.size(-1)
        
        return -0.5 * (mahalanobis_term + self.log_det_Sigma + k * math.log(2 * math.pi))

    def sample(self, shape: int | tuple) -> Tensor:
        """
        Samples from a general multivariate Gaussian distribution.
        shape: Tuple representing the shape of the samples (batch_size, d).
        """
        z = torch.randn(shape)  # Standard normal samples
        L = torch.cholesky(self.Sigma)  # Cholesky decomposition of Sigma (for sampling)
        return self.mu + z @ L.T  # Transform standard normal samples

class BetaSource(SourceDistribution):
    def __init__(self, alpha: float, beta: float, lowerbound: float = 0., upperbound: float = 1., **kwargs):
        """
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta
        """
        self.alpha = alpha
        self.beta = beta
        self.dist = torch.distributions.Beta(alpha, beta)
        # per default, the beta distribution is defined on the interval [0, 1]
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def log_prob(self, x: Tensor) -> Tensor:
        """Will return nan if x is outside the support interval [lowerbound, upperbound]"""
        lp = self.dist.log_prob(x).sum(dim=-1)
        # lp = log_beta(x, self.alpha, self.beta)
        # Transform log-probabilities to the desired interval
        # TODO: Check if this is correct
        return lp - math.log(self.upperbound - self.lowerbound)

    def sample(self, shape: int | tuple) -> Tensor:
        samples = self.dist.sample(tuple(shape))
        # Transform samples to the desired interval
        return samples * (self.upperbound - self.lowerbound) + self.lowerbound

def log_mixture_of_gaussians(z: Tensor, mus: List[Tensor], sigmas: List[Tensor], pis: List[float]) -> Tensor:
    # compute the probability for each component and then sum them according to the mixture weights
    log_probs = []
    for mu, sigma, pi in zip(mus, sigmas, pis):
        # Compute log-probability for each Gaussian component
        log_prob = -0.5 * (((z - mu) / sigma).pow(2) + 2 * sigma.log() + math.log(2 * math.pi)).sum(dim=-1)
        log_probs.append(log_prob + math.log(pi))  # Add log of mixture weight
    
    # Log-sum-exp for stability when summing probabilities in log-space
    return torch.logsumexp(torch.stack(log_probs, dim=-1), dim=-1)

class UniformSource(SourceDistribution):
    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        self.low = low
        self.high = high

    def log_prob(self, x: Tensor) -> Tensor:
        return torch.zeros(x.size(0))

    def sample(self, shape: int | tuple) -> Tensor:
        return torch.rand(shape) * (self.high - self.low) + self.low

class MixtureOfGaussians(SourceDistribution):
    """
    # Define your mixture of Gaussians (K components)
    mus = [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 5.0])]  # Means for each component
    sigmas = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]  # Std deviations for each component
    pis = [0.5, 0.5]  # Equal mixture weights
    """
    def __init__(self, mus: List[Tensor], sigmas: List[Tensor], pis: List[float], **kwargs):
        self.mus = mus  # List of means for each component
        self.sigmas = sigmas  # List of standard deviations (or covariance matrices)
        self.pis = pis  # List of mixture weights (sums to 1)
        self.K = len(mus)  # Number of components
    
    def log_prob(self, z: Tensor) -> Tensor:
        return log_mixture_of_gaussians(z, self.mus, self.sigmas, self.pis)

    def sample(self, shape: Tuple[int]):
        # Sample component index based on mixture weights
        component = torch.multinomial(torch.tensor(self.pis), shape[0], replacement=True)
        
        # For each sample, draw from the corresponding Gaussian component
        samples = []
        for i in range(shape[0]):
            mu = self.mus[component[i]]
            sigma = self.sigmas[component[i]]
            samples.append(torch.normal(mu, sigma))
        
        return torch.stack(samples)


_distributions = {
    "normal": StandardNormalSource,
    "isotropic": IsotropicGaussianSource,
    "diagonal": DiagonalGaussianSource,
    "gaussian": GaussianSource,
    "beta": BetaSource,
    "mog": MixtureOfGaussians,
}

def get_source_distribution(type: str = "gaussian", **kwargs):
    assert type in _distributions, \
        f"Unknown source distribution: {type}.\n Try one of {list(_distributions.keys())}"
    return _distributions[type.lower()](**kwargs)
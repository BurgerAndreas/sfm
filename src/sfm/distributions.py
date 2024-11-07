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


class SourceDistribution:
    def __init__(self, **kwargs):
        pass

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute the log probability of the given samples under the distribution.

        Args:
            x (torch.Tensor): Input samples.

        Returns:
            torch.Tensor: Log probabilities of the input samples.
        """
        return None

    def sample(self, shape: int | tuple) -> Tensor:
        """
        Draw samples from the distribution.

        Args:
            shape (int or tuple): Shape of the samples to draw.

        Returns:
            torch.Tensor: Samples drawn from the distribution.
        """
        raise NotImplementedError


class UniformSource(SourceDistribution):
    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        self.dist = torch.distributions.Uniform(low, high)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


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
    mahalanobis_distance = (
        (diff.unsqueeze(-1).transpose(-1, -2) @ inv_Sigma @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    )
    # Compute the log determinant of the covariance matrix
    log_det_Sigma = torch.logdet(Sigma)
    # Number of dimensions
    k = x.size(-1)
    # Log probability calculation
    return -0.5 * (mahalanobis_distance + log_det_Sigma + k * math.log(2 * math.pi))


def log_beta(x: Tensor, alpha: float, beta: float) -> Tensor:
    beta_func = (
        torch.lgamma(torch.tensor(alpha)) + torch.lgamma(torch.tensor(beta)) - torch.lgamma(torch.tensor(alpha + beta))
    )
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
        Diagonal Gaussian distribution is a special case of the
        multivariate Gaussian distribution where the covariance matrix is diagonal.
        Meaning that the dimensions are uncorrelated.
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
        squared_term = (((x - self.mu) ** 2) / (self.sigma**2)).sum(dim=-1)
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
        mahalanobis_term = (
            (diff.unsqueeze(-1).transpose(-1, -2) @ self.inv_Sigma @ diff.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        )

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


def log_mixture_of_gaussians(z: Tensor, mus: List[Tensor], sigmas: List[Tensor], pis: List[float]) -> Tensor:
    # compute the probability for each component and then sum them according to the mixture weights
    log_probs = []
    for mu, sigma, pi in zip(mus, sigmas, pis):
        # Compute log-probability for each Gaussian component
        log_prob = -0.5 * (((z - mu) / sigma).pow(2) + 2 * sigma.log() + math.log(2 * math.pi)).sum(dim=-1)
        log_probs.append(log_prob + math.log(pi))  # Add log of mixture weight

    # Log-sum-exp for stability when summing probabilities in log-space
    return torch.logsumexp(torch.stack(log_probs, dim=-1), dim=-1)


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


class BetaSource(SourceDistribution):
    def __init__(self, alpha: float, beta: float, **kwargs):
        """
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta
        """
        self.alpha = alpha
        self.beta = beta
        self.dist = torch.distributions.Beta(alpha, beta)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class CauchySource(SourceDistribution):
    def __init__(self, loc: float, scale: float, **kwargs):
        """Symmetric, bell-shaped like the Gaussian distribution, but with heavier tails."""
        self.dist = torch.distributions.Cauchy(loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class Chi2Source(SourceDistribution):
    def __init__(self, df: float, **kwargs):
        """defined as the sum of the squares of kk independent standard normal random variables.
        For low k values, the distribution is right-skewed, while for high k values, it approaches a normal distribution.
        """
        self.dist = torch.distributions.Chi2(df)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


# class ContinuousBernoulliSource(SourceDistribution):
#     def __init__(self, logits: Tensor, **kwargs):
#         """Continuous Bernoulli is diffrent from the RelaxedBernoulli distribution
#         in that it is being bounded between 0 and 1.
#         PDF) that is piecewise linear, with a peak at either 0 or 1 depending on the value of λ.
#         """
#         self.dist = torch.distributions.ContinuousBernoulli(logits=logits)

#     def log_prob(self, x: Tensor) -> Tensor:
#         return self.dist.log_prob(x)

#     def sample(self, shape: int | tuple) -> Tensor:
#         return self.dist.sample(shape)


class DirichletSource(SourceDistribution):
    def __init__(self, concentration: Tensor, **kwargs):
        """Dirichlet distribution is a multivariate generalization of the Beta distribution.
        Meaning the dimensions are correlated.
        """
        self.dist = torch.distributions.Dirichlet(concentration)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class ExponentialSource(SourceDistribution):
    def __init__(self, rate: float, **kwargs):
        self.dist = torch.distributions.Exponential(rate)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class FisherSnedecorSource(SourceDistribution):
    def __init__(self, df1: float, df2: float, **kwargs):
        """Fisher-Snedecor distribution, often simply referred to as the F-distribution, is a continuous probability distribution that arises frequently in the analysis of variance (ANOVA) and other statistical tests.
        defined as the ratio of two chi-square distributions, each divided by their respective degrees of freedom.
        right-skewed and approaches symmetry as the degrees of freedom increase.
        """
        self.dist = torch.distributions.FisherSnedecor(df1, df2)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class GammaSource(SourceDistribution):
    def __init__(self, concentration: float, rate: float, **kwargs):
        self.dist = torch.distributions.Gamma(concentration, rate)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class GumbelSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, **kwargs):
        """used in extreme value theory.
        It is often used to model the distribution of the maximum (or minimum)
        of a number of samples from the same distribution,
        such as the highest temperature in a year or the largest flood in a century."""
        self.dist = torch.distributions.Gumbel(loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class HalfCauchySource(SourceDistribution):
    def __init__(self, scale: float, **kwargs):
        self.dist = torch.distributions.HalfCauchy(scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class HalfNormalSource(SourceDistribution):
    def __init__(self, scale: float, **kwargs):
        """HalfNormal distribution is a distribution of the absolute value of a random variable."""
        self.dist = torch.distributions.HalfNormal(scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class InverseGammaSource(SourceDistribution):
    def __init__(self, concentration: float, rate: float, **kwargs):
        """InverseGamma distribution describes the distribution of the reciprocal of a random variable.
        Used in Bayesian statistics, where the distribution arises as the marginal posterior distribution for the unknown variance of a normal distribution, if an uninformative prior is used.
        Domain is x > 0."""
        self.dist = torch.distributions.InverseGamma(concentration, rate)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class KumaraswamySource(SourceDistribution):
    def __init__(self, a: float, b: float, **kwargs):
        """Kumaraswamy distribution is a continuous probability distribution defined on the interval [0, 1].
        Similar to the Beta distribution, but it's
        probability density function, cumulative distribution function and quantile functions can be expressed in closed form.
        """
        self.dist = torch.distributions.Kumaraswamy(a, b)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class LKJCholeskySource(SourceDistribution):
    def __init__(self, dim: int, concentration: float = 1.0, **kwargs):
        """LKJ distribution describes the distribution of the Cholesky factor of a correlation matrix.
        For example in the real world, the distribution of the correlation matrix of stock returns.
        In the context of neural networks, the distribution of the correlation matrix of the weights.
        The distribution is parameterized by the dimensionality of the matrix and a concentration parameter.
        The domain of the concentration parameter is (0, ∞) and controls the amount of correlation."""
        self.dist = torch.distributions.LKJCholesky(dim, concentration)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class LaplaceSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, **kwargs):
        """Laplacian distribution is a distribution of the difference between two independent exponential random variables.
        For example, in the context of neural networks, the distribution of the difference between two weights.
        """
        self.dist = torch.distributions.Laplace(loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class LogNormalSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, **kwargs):
        """Describes the distribution of the logarithm of a random variable.
        Useful for modeling quantities that are strictly positive."""
        self.dist = torch.distributions.LogNormal(loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class LowRankMultivariateNormalSource(SourceDistribution):
    def __init__(self, loc: Tensor, cov_factor: Tensor, cov_diag: Tensor, **kwargs):
        """Describes the distribution of a multivariate Gaussian with a low-rank covariance matrix.
        For example, in the context of neural networks, the distribution of the weights.
        Low-rank covariance matrix is a sum of a rank-r matrix and a diagonal matrix."""
        self.dist = torch.distributions.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class MultivariateNormalSource(SourceDistribution):
    def __init__(self, loc: Tensor, covariance_matrix: Tensor, **kwargs):
        """Multivariate = multiple dimensions, normal = Gaussian distribution.
        Covariance matrix describes the relationship between the dimensions.
        """
        self.dist = torch.distributions.MultivariateNormal(loc, covariance_matrix)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class NormalSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, **kwargs):
        """Describes Gaussian distribution with mean=loc and std=scale,
        where all dimensions are uncorrelated and have the same mean and variance."""
        self.dist = torch.distributions.Normal(loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class ParetoSource(SourceDistribution):
    def __init__(self, scale: float, alpha: float, **kwargs):
        """Pareto distribution is a heavy tail characterized by the Pareto principle, also known as the 80/20 rule."""
        self.dist = torch.distributions.Pareto(scale, alpha)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class RelaxedBernoulliSource(SourceDistribution):
    def __init__(self, logits: Tensor, temperature: float, **kwargs):
        """RelaxedBernoulli distribution is a continuous relaxation of the Bernoulli distribution,
        relaxing the discrete Bernoulli distribution to a continuous distribution.
        Temperature controls the degree of relaxation."""
        self.dist = torch.distributions.RelaxedBernoulli(temperature, logits=logits)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


# class LogitRelaxedBernoulliSource(SourceDistribution):
#     def __init__(self, logits: Tensor, temperature: float, **kwargs):
#         """LogitRelaxedBernoulli distribution is a continuous relaxation of the Bernoulli distribution,
#         given by the logistic sigmoid of the Bernoulli logits.
#         Logits could come from a neural network output, for example."""
#         self.dist = torch.distributions.LogitRelaxedBernoulli(temperature, logits=logits)

#     def log_prob(self, x: Tensor) -> Tensor:
#         return self.dist.log_prob(x)

#     def sample(self, shape: int | tuple) -> Tensor:
#         return self.dist.sample(shape)

# class RelaxedOneHotCategoricalSource(SourceDistribution):
#     def __init__(self, logits: Tensor, temperature: float, **kwargs):
#         """Describes the distribution of a one-hot categorical variable."""
#         self.dist = torch.distributions.RelaxedOneHotCategorical(temperature, logits=logits)

#     def log_prob(self, x: Tensor) -> Tensor:
#         return self.dist.log_prob(x)

#     def sample(self, shape: int | tuple) -> Tensor:
#         return self.dist.sample(shape)


class StudentTSource(SourceDistribution):
    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0, **kwargs):
        """Describes the distribution of the difference between two independent Gaussian random variables."""
        self.dist = torch.distributions.StudentT(df, loc, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


# class TransformedDistributionSource(SourceDistribution):
#     def __init__(self, base_distribution, transform, **kwargs):
#         """Describe a distribution that is transformed by a function."""
#         self.dist = torch.distributions.TransformedDistribution(base_distribution, transform)

#     def log_prob(self, x: Tensor) -> Tensor:
#         return self.dist.log_prob(x)

#     def sample(self, shape: int | tuple) -> Tensor:
#         return self.dist.sample(shape)


class VonMisesSource(SourceDistribution):
    def __init__(self, loc: float, concentration: float, **kwargs):
        """Describes the distribution of angles on the unit circle."""
        self.dist = torch.distributions.VonMises(loc, concentration)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class WeibullSource(SourceDistribution):
    def __init__(self, scale: float, concentration: float, **kwargs):
        """Weibull distribution describes the distribution of the minimum of a set of random variables."""
        self.dist = torch.distributions.Weibull(scale, concentration)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


class WishartSource(SourceDistribution):
    def __init__(self, df: float, scale: Tensor, **kwargs):
        """Describe the distribution of a positive semi-definite matrix."""
        self.dist = torch.distributions.Wishart(df, scale)

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, shape: int | tuple) -> Tensor:
        return self.dist.sample(shape)


_distributions = {
    "normal": StandardNormalSource,
    "gaussian": GaussianSource,
    "isotropic": IsotropicGaussianSource,
    "diagonal": DiagonalGaussianSource,
    "beta": BetaSource,
    "mog": MixtureOfGaussians,
    "cauchy": CauchySource,
    "chi2": Chi2Source,
    "dirichlet": DirichletSource,
    "exponential": ExponentialSource,
    "fisher": FisherSnedecorSource,
    "gamma": GammaSource,
    "gumbel": GumbelSource,
    "halfcauchy": HalfCauchySource,
    "halfnormal": HalfNormalSource,
    "inversegamma": InverseGammaSource,
    "kumaraswamy": KumaraswamySource,
    "lkj": LKJCholeskySource,
    "laplace": LaplaceSource,
    "lognormal": LogNormalSource,
    "lowrank": LowRankMultivariateNormalSource,
    "multivariate": MultivariateNormalSource,
    "pareto": ParetoSource,
    "relaxedbernoulli": RelaxedBernoulliSource,
    "studentt": StudentTSource,
    "vonmises": VonMisesSource,
    "weibull": WeibullSource,
    "wishart": WishartSource,
}


def get_source_distribution(type: str = "gaussian", **kwargs):
    assert type in _distributions, f"Unknown source distribution: {type}.\n Try one of {list(_distributions.keys())}"
    return _distributions[type.lower()](**kwargs)

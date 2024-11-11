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
    def __init__(self, data_dim: int = 2, device: str = "cpu", **kwargs):
        self.data_dim = data_dim
        self.device = device
        self.dist = None

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute the log probability of the given samples under the distribution.

        Args:
            x (torch.Tensor): Input samples.

        Returns:
            torch.Tensor: Log probabilities of the input samples.
        """
        return self.dist.log_prob(x).sum(dim=-1)

    def sample(self, nsamples: int | tuple) -> Tensor:
        """
        Draw samples from the distribution.

        Args:
            shape (int or tuple): Shape of the samples to draw.

        Returns:
            torch.Tensor: Samples drawn from the distribution.
        """
        return self.dist.sample((nsamples, self.data_dim))


class CoupledSourceDistribution(SourceDistribution):
    """Distributions where the data dimensions are coupled (not independent).
    The log-probability and the samples have different shapes.
    """

    def log_prob(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x)

    def sample(self, nsamples: int | tuple) -> Tensor:
        return self.dist.sample((nsamples,))


class UniformSource(SourceDistribution):
    def __init__(self, low: float = -1.0, high: float = 1.0, data_dim: int = 2, **kwargs):
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Uniform(low, high)


class StandardNormalSource(SourceDistribution):
    def __init__(self, data_dim: int = 2, mean:float = 0, std: float = 0.3, **kwargs):
        """Mean=0, std=1"""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Normal(mean, std)

    # def log_prob(self, x: Tensor) -> Tensor:
    #     return self.dist.log_prob(x).sum(dim=-1)

    # should be the same as
    # def log_normal(x: Tensor) -> Tensor:
    #     return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2


class IsotropicGaussianSource(CoupledSourceDistribution):
    def __init__(self, mu: Tensor = torch.tensor([0.0, 0.0]), sigma: float = 1.0, data_dim: int = 2, **kwargs):
        """Isotropic Gaussian with mean=mu and std=sigma.
        Isotropic means that the covariance matrix is a multiple of the identity matrix.
        """
        super().__init__(data_dim, **kwargs)
        self.mu = torch.as_tensor(mu)
        self.sigma = sigma
        self.dist = torch.distributions.MultivariateNormal(
            loc=self.mu, covariance_matrix=torch.diag(torch.tensor([sigma] * data_dim))
        )


class DiagonalGaussianSource(CoupledSourceDistribution):
    def __init__(self, mu: Tensor, sigma: Tensor, data_dim: int = 2, **kwargs):
        """
        Diagonal Gaussian distribution is a special case of the
        multivariate Gaussian distribution where the covariance matrix is diagonal.
        Meaning that the dimensions are uncorrelated.
        mu: Tensor of shape (d,) where d is the dimensionality of the distribution
        sigma: Tensor of shape (d,) representing the standard deviation for each dimension
        """
        super().__init__(data_dim, **kwargs)
        self.mu = torch.as_tensor(mu)
        self.sigma = torch.as_tensor(sigma)
        self.dist = torch.distributions.MultivariateNormal(loc=self.mu, covariance_matrix=torch.diag(self.sigma))


class GaussianSource(CoupledSourceDistribution):
    def __init__(self, mu: Tensor, Sigma: Tensor, data_dim: int = 2, **kwargs):
        """Gaussian distribution with mean=mu and covariance matrix=Sigma.
        In contrast to the DiagonalGaussianSource, this distribution allows for correlations between dimensions.
        mu: Tensor of shape (d,) where d is the dimensionality of the distribution.
        Sigma: Tensor of shape (d, d), the covariance matrix.
        """
        super().__init__(data_dim, **kwargs)
        mu = torch.as_tensor(mu)
        Sigma = torch.as_tensor(Sigma)
        assert Sigma.shape == (data_dim, data_dim), f"Sigma has wrong shape {Sigma.shape}"
        self.dist = torch.distributions.MultivariateNormal(mu, Sigma)


class MixtureOfGaussians(CoupledSourceDistribution):
    """
    # Define your mixture of Gaussians (K components)
    mus = [torch.tensor([0.0, 0.0]), torch.tensor([5.0, 5.0])]  # Means for each component
    sigmas = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]  # Std deviations for each component
    pis = [0.5, 0.5]  # Equal mixture weights
    """

    def __init__(self, mus: Tensor, sigmas: Tensor, pis: Tensor, data_dim: int = 2, **kwargs):
        # means (components x dimensions)
        super().__init__(data_dim, **kwargs)
        mus = torch.as_tensor(mus)
        # std deviations (components x dimensions)
        sigmas = torch.as_tensor(sigmas)
        # mixture weights (components)
        pis = torch.as_tensor(pis)
        mix = torch.distributions.Categorical(pis)
        self.K = len(mus)  # Number of components
        assert mus.shape == (self.K, data_dim), f"mus has wrong shape {mus.shape}"
        assert sigmas.shape == (self.K, data_dim), f"sigmas has wrong shape {sigmas.shape}"
        assert pis.shape == (self.K,), f"pis has wrong shape {pis.shape}"
        assert all(0 <= pi <= 1 for pi in pis), "all pis must be between 0 and 1"
        assert sum(pis) == 1, "pis must sum to 1"
        comp = torch.distributions.Independent(torch.distributions.Normal(mus, sigmas), 1)
        self.dist = torch.distributions.MixtureSameFamily(mix, comp)


class BetaSource(SourceDistribution):
    def __init__(self, alpha: float, beta: float, data_dim: int = 2, **kwargs):
        """
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.beta.Beta
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Beta(
            torch.as_tensor(alpha).to(self.device), torch.as_tensor(beta).to(self.device)
        )

    def to(self, device):
        self.dist._dirichlet.concentration = self.dist._dirichlet.concentration.to(device)
        return self


class CauchySource(SourceDistribution):
    def __init__(self, loc: float, scale: float, data_dim: int = 2, **kwargs):
        """Symmetric, bell-shaped like the Gaussian distribution, but with heavier tails."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Cauchy(loc, scale)


class Chi2Source(SourceDistribution):
    def __init__(self, df: float, data_dim: int = 2, **kwargs):
        """defined as the sum of the squares of kk independent standard normal random variables.
        For low k values, the distribution is right-skewed, while for high k values, it approaches a normal distribution.
        """
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Chi2(df)


class DirichletSource(CoupledSourceDistribution):
    def __init__(self, concentration: Tensor, data_dim: int = 2, **kwargs):
        """Dirichlet distribution is a multivariate generalization of the Beta distribution.
        Meaning the dimensions are correlated.
        """
        super().__init__(data_dim, **kwargs)
        concentration = torch.as_tensor(concentration)
        assert concentration.shape == (data_dim,), f"Concentration has wrong shape {concentration.shape}"
        self.dist = torch.distributions.Dirichlet(concentration)


class ExponentialSource(SourceDistribution):
    def __init__(self, rate: float, data_dim: int = 2, **kwargs):
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Exponential(rate)


class FisherSnedecorSource(SourceDistribution):
    def __init__(self, df1: float, df2: float, data_dim: int = 2, **kwargs):
        """Fisher-Snedecor distribution, often simply referred to as the F-distribution, is a continuous probability distribution that arises frequently in the analysis of variance (ANOVA) and other statistical tests.
        defined as the ratio of two chi-square distributions, each divided by their respective degrees of freedom.
        right-skewed and approaches symmetry as the degrees of freedom increase.
        """
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.FisherSnedecor(df1, df2)


class GammaSource(SourceDistribution):
    def __init__(self, concentration: float, rate: float, data_dim: int = 2, **kwargs):
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Gamma(concentration, rate)


class GumbelSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, data_dim: int = 2, **kwargs):
        """used in extreme value theory.
        It is often used to model the distribution of the maximum (or minimum)
        of a number of samples from the same distribution,
        such as the highest temperature in a year or the largest flood in a century."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Gumbel(loc, scale)


class HalfCauchySource(SourceDistribution):
    def __init__(self, scale: float, data_dim: int = 2, **kwargs):
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.HalfCauchy(scale)


class HalfNormalSource(SourceDistribution):
    def __init__(self, scale: float, data_dim: int = 2, **kwargs):
        """HalfNormal distribution is a distribution of the absolute value of a random variable."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.HalfNormal(scale)


class InverseGammaSource(SourceDistribution):
    def __init__(self, concentration: float, rate: float, data_dim: int = 2, **kwargs):
        """InverseGamma distribution describes the distribution of the reciprocal of a random variable.
        Used in Bayesian statistics, where the distribution arises as the marginal posterior distribution for the unknown variance of a normal distribution, if an uninformative prior is used.
        Domain is x > 0."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.InverseGamma(concentration, rate)


class KumaraswamySource(SourceDistribution):
    def __init__(self, a: float, b: float, data_dim: int = 2, **kwargs):
        """Kumaraswamy distribution is a continuous probability distribution defined on the interval [0, 1].
        Similar to the Beta distribution, but it's
        probability density function, cumulative distribution function and quantile functions can be expressed in closed form.
        """
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Kumaraswamy(a, b)


class LaplaceSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, data_dim: int = 2, **kwargs):
        """Laplacian distribution is a distribution of the difference between two independent exponential random variables.
        For example, in the context of neural networks, the distribution of the difference between two weights.
        """
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Laplace(loc, scale)


class LogNormalSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, data_dim: int = 2, **kwargs):
        """Describes the distribution of the logarithm of a random variable.
        Useful for modeling quantities that are strictly positive."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.LogNormal(loc, scale)


class LowRankMultivariateNormalSource(CoupledSourceDistribution):
    def __init__(self, loc: Tensor, cov_factor: Tensor, cov_diag: Tensor, data_dim: int = 2, **kwargs):
        """Describes the distribution of a multivariate Gaussian with a low-rank covariance matrix.
        For example, in the context of neural networks, the distribution of the weights.
        Low-rank covariance matrix is a sum of a rank-r matrix and a diagonal matrix."""
        super().__init__(data_dim, **kwargs)
        self.loc = torch.as_tensor(loc)
        self.cov_factor = torch.as_tensor(cov_factor)
        self.cov_diag = torch.as_tensor(cov_diag)
        self.dist = torch.distributions.LowRankMultivariateNormal(self.loc, self.cov_factor, self.cov_diag)


class MultivariateNormalSource(CoupledSourceDistribution):
    def __init__(self, loc: Tensor, covariance_matrix: Tensor, data_dim: int = 2, **kwargs):
        """Multivariate = multiple dimensions, normal = Gaussian distribution.
        Covariance matrix describes the relationship between the dimensions.
        """
        super().__init__(data_dim, **kwargs)
        self.loc = torch.as_tensor(loc)
        covariance_matrix = torch.as_tensor(covariance_matrix)
        assert covariance_matrix.shape == (
            data_dim,
            data_dim,
        ), f"Covariance matrix has wrong shape {covariance_matrix.shape}"
        self.dist = torch.distributions.MultivariateNormal(self.loc, covariance_matrix)


class NormalSource(SourceDistribution):
    def __init__(self, loc: float, scale: float, data_dim: int = 2, **kwargs):
        """Describes Gaussian distribution with mean=loc and std=scale,
        where all dimensions are uncorrelated and have the same mean and variance."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Normal(loc, scale)


class ParetoSource(SourceDistribution):
    def __init__(self, scale: float, alpha: float, data_dim: int = 2, **kwargs):
        """Pareto distribution is a heavy tail characterized by the Pareto principle, also known as the 80/20 rule."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Pareto(scale, alpha)


class RelaxedBernoulliSource(SourceDistribution):
    def __init__(self, logits: Tensor, temperature: Tensor, data_dim: int = 2, **kwargs):
        """RelaxedBernoulli distribution is a continuous relaxation of the Bernoulli distribution,
        relaxing the discrete Bernoulli distribution to a continuous distribution.
        Temperature controls the degree of relaxation."""
        super().__init__(data_dim, **kwargs)
        self.logits = torch.as_tensor(logits)
        self.temperature = torch.as_tensor(temperature)
        assert self.logits.shape == (data_dim,), f"Logits have wrong shape {self.logits.shape}"
        # assert self.temperature.shape == (data_dim,), f"Temperature has wrong shape {self.temperature.shape}"
        self.dist = torch.distributions.RelaxedBernoulli(temperature=self.temperature, logits=self.logits)

    def sample(self, nsamples: int | tuple) -> Tensor:
        return self.dist.sample((nsamples,))


class StudentTSource(SourceDistribution):
    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0, data_dim: int = 2, **kwargs):
        """Describes the distribution of the difference between two independent Gaussian random variables."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.StudentT(df, loc, scale)


class VonMisesSource(SourceDistribution):
    def __init__(self, loc: float, concentration: float, data_dim: int = 2, **kwargs):
        """Describes the distribution of angles on the unit circle."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.VonMises(loc, concentration)


class WeibullSource(SourceDistribution):
    def __init__(self, scale: float, concentration: float, data_dim: int = 2, **kwargs):
        """Weibull distribution describes the distribution of the minimum of a set of random variables."""
        super().__init__(data_dim, **kwargs)
        self.dist = torch.distributions.Weibull(scale, concentration)


_distributions = {
    "uniform": UniformSource,
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
    "laplace": LaplaceSource,
    "lognormal": LogNormalSource,
    "lowrank": LowRankMultivariateNormalSource,
    "multivariate": MultivariateNormalSource,
    "pareto": ParetoSource,
    "relaxedbernoulli": RelaxedBernoulliSource,
    "studentt": StudentTSource,
    "vonmises": VonMisesSource,
    "weibull": WeibullSource,
}


def get_source_distribution(trgt: str = "gaussian", **kwargs):
    assert trgt in _distributions, f"Unknown source distribution: {trgt}.\n Try one of {list(_distributions.keys())}"
    return _distributions[trgt.lower()](**kwargs)

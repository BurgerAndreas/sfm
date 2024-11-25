# Gaussian Mixture Model

import numpy as np
import torch
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from enum import Enum
from dataclasses import dataclass
from sfm.plotstyle import set_seaborn_style, set_style_after

class CovarianceType(Enum):
    FULL = 'full'           # Full covariance matrix
    DIAGONAL = 'diagonal'   # Diagonal covariance matrix
    SPHERICAL = 'spherical' # Single variance parameter per component
    TIED = 'tied'          # Share same covariance matrix across components


class GaussianMixture():
    def __init__(
        self, 
        n_components=2, 
        max_iter=100, 
        tol=1e-5, 
        init_strategy='kmeans',
        trgtdist=None,
        dtype=torch.float32,
        device='cpu',
        data_dim=2,
        **kwargs,
    ):
        self.dtype = dtype
        self.device = device
        self.data_dim = data_dim
        # GMM parameters
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_strategy = init_strategy
        maxsamples = 100
        if hasattr(trgtdist, "trainset"):
            # 0 because MNIST also returns class labels
            # [:10000]
            all_data = torch.stack([data[0] for data in trgtdist.trainset], dim=0)
            all_data = all_data[:maxsamples]
        else:
            # sample from target distribution
            all_data = trgtdist.sample(maxsamples)
        self.fit(all_data)
        
    def initialize_parameters(self, X):
        X = X.to(self.device)
        # if data is [B, C, H, W], flatten it to [B, C*H*W]
        if len(X.shape) > 2:
            X = X.view(X.shape[0], -1)
        n_samples, n_features = X.shape
        
        if self.init_strategy == 'kmeans':
            # Use kmeans++ initialization for better starting points
            # [n_components, D]
            self.means = self._kmeans_plus_plus(X, self.n_components).to(self.device)
        else:
            # Random initialization
            random_idx = torch.randperm(n_samples, device=self.device)[:self.n_components]
            # [n_components, D]
            self.means = X[random_idx].to(self.device)
        
        # Initialize covariances with identity matrix scaled by data variance
        # [n_components, D, D]
        data_var = torch.var(X, dim=0).mean()
        if data_var == 0:
            data_var = 1.0  # Fallback if variance is 0
        self.covs = torch.stack([
            torch.eye(n_features, device=self.device) * data_var
            for _ in range(self.n_components)
        ])
        
        # Initialize mixing coefficients
        self.weights = torch.ones(self.n_components, device=self.device) / self.n_components
        
        # Initialize responsibilities
        self.responsibilities = torch.zeros((n_samples, self.n_components), device=self.device)
        
    def _kmeans_plus_plus(self, X, k):
        """Initialize cluster centers using k-means++ algorithm"""
        X = X.to(self.device)
        n_samples, n_features = X.shape
        centers = torch.zeros((k, n_features), device=self.device)
        
        # Choose first center randomly
        centers[0] = X[torch.randint(n_samples, (1,), device=X.device)]
        
        # Choose remaining centers
        for i in range(1, k):
            # Compute distances to closest centers
            distances = torch.min(torch.stack([
                torch.sum((X - center) ** 2, dim=1)
                for center in centers[:i]
            ]), dim=0)[0]
            
            # Add small epsilon to avoid zero distances
            distances = distances + 1e-10
            
            # Choose next center with probability proportional to distance squared
            probs = distances / distances.sum()
            centers[i] = X[torch.multinomial(probs, 1)]
            
        return centers
        
    def expectation_step(self, X):
        n_samples = X.shape[0]
        weighted_likelihoods = torch.zeros((n_samples, self.n_components), device=self.device)
        
        for k in range(self.n_components):
            try:
                weighted_likelihoods[:, k] = self.weights[k] * torch.exp(
                    torch.distributions.MultivariateNormal(
                        self.means[k], self.covs[k]
                    ).log_prob(X)
                )
            except RuntimeError:
                # If covariance matrix is singular, add small diagonal term
                self.covs[k] += torch.eye(X.shape[1]) * 1e-6
                weighted_likelihoods[:, k] = self.weights[k] * torch.exp(
                    torch.distributions.MultivariateNormal(
                        self.means[k], self.covs[k]
                    ).log_prob(X)
                )
            
            assert weighted_likelihoods[:, k].isfinite().all(), f"weighted likelihoods contain NaN or Inf"
            assert self.covs[k].isfinite().all(), f"covariance matrix contains NaN or Inf"
            
        # Normalize responsibilities
        total_likelihood = weighted_likelihoods.sum(dim=1, keepdim=True)
        assert total_likelihood.isfinite().all(), f"total likelihood contains NaN or Inf"
        assert (total_likelihood > 0).all(), f"total likelihood is not positive"
        
        self.responsibilities = weighted_likelihoods / total_likelihood
        assert self.responsibilities.isfinite().all(), f"responsibilities contain NaN or Inf"
        
        return torch.sum(torch.log(total_likelihood))
    
    def maximization_step(self, X):
        n_samples = X.shape[0]
        
        # Update weights
        Nk = self.responsibilities.sum(dim=0)
        self.weights = Nk / n_samples
        
        # Update means
        self.means = torch.matmul(self.responsibilities.T, X) / Nk.unsqueeze(1).to(self.device)
        assert self.means.isfinite().all(), f"means contain NaN or Inf"
        
        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            # self.covs[k] = torch.matmul(
            #     (self.responsibilities[:, k].unsqueeze(1) * diff).T, diff
            # ) / Nk[k]
            weighted_diff = self.responsibilities[:, k].unsqueeze(1) * diff
            self.covs[k] = torch.matmul(weighted_diff.T, diff) / Nk[k]
            
            # Add regularization to ensure positive definiteness
            min_var = 1e-6
            self.covs[k] = self.covs[k] + torch.eye(X.shape[1], device=self.device) * min_var
            
        self.covs = self.covs.to(self.device)
        assert self.covs.isfinite().all(), f"covariance matrices contain NaN or Inf"
            
    def fit(self, X):
        """Fit the model to data X of shape [B, D]"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        
        X = X.to(self.device)
        # reshape x [B, 1, 28, 28] -> [B, D]
        X = X.view(X.shape[0], -1)
        
        self.n_features = X.shape[1]
        self.initialize_parameters(X)
        
        log_likelihood = float('-inf')
        
        for iteration in range(self.max_iter):
            # E-step
            new_log_likelihood = self.expectation_step(X)
            
            # M-step
            self.maximization_step(X)
            
            # Check convergence
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
                
            log_likelihood = new_log_likelihood
            
        return self
    
    def predict(self, X):
        """Return the most likely component for each sample"""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.to(self.device)
        n_samples = X.shape[0]
        likelihoods = torch.zeros((n_samples, self.n_components), device=self.device)
        
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights[k] * torch.exp(
                torch.distributions.MultivariateNormal(
                    self.means[k], self.covs[k]
                ).log_prob(X)
            )
            
        return torch.argmax(likelihoods, dim=1)
    
    def sample(self, n_samples) -> torch.Tensor:
        """Generate samples from the fitted mixture model.
        Returns [n_samples, D]
        """
        # Choose components based on weights
        components = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # Generate samples from chosen components
        samples = torch.zeros((n_samples, self.n_features), device=self.device)
        for k in range(self.n_components):
            mask = components == k
            samples[mask.to(self.device)] = torch.distributions.MultivariateNormal(
                self.means[k], self.covs[k]
            ).sample((mask.sum(),))
            
        return samples
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of the given samples under the distribution.

        Args:
            x (torch.Tensor): Input samples.

        Returns:
            torch.Tensor: Log probabilities of the input samples.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            
        # reshape x [B, 1, 28, 28] -> [B, D]
        x = x.view(x.shape[0], -1)
        
        x = x.to(self.device)
        log_probs = torch.zeros(x.shape[0], self.n_components, device=self.device)
        for k in range(self.n_components):
            log_probs[:, k] = torch.log(self.weights[k]) \
                + torch.distributions.MultivariateNormal(
                    self.means[k], self.covs[k]
                ).log_prob(x)
        
        return torch.logsumexp(log_probs, dim=1)
    
    # def log_prob(self, X: torch.Tensor) -> torch.Tensor:
    #     n_samples = X.shape[0]
    #     log_weights = torch.log(self.weights)
    #     log_probs = torch.zeros(n_samples, self.n_components, device=X.device)
        
    #     for k in range(self.n_components):
    #         mvn = MultivariateNormal(self.means[k], self.covs[k])
    #         log_probs[:, k] = log_weights[k] + mvn.log_prob(X)
        
    #     return torch.logsumexp(log_probs, dim=1)
    
    def visualize(self, X, labels=None, title="Gaussian Mixture Model"):
        """Visualize the data and fitted model"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if X.shape[1] == 2:
            return self._plot_2d(X, labels, title)
        else:
            return self._plot_high_dim(X, labels, title)
    
    def _plot_2d(self, X, labels=None, title="2D Gaussian Mixture"):
        """Plot 2D data and model"""
        set_seaborn_style()
        fig = plt.figure(figsize=(10, 8))
        
        # Plot data points
        if labels is None:
            labels = self.predict(X).cpu().numpy()
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # Plot means
        means_np = self.means.detach().cpu().numpy()
        plt.scatter(means_np[:, 0], means_np[:, 1], 
                   c='red', marker='x', s=200, linewidth=3,
                   label='Cluster Centers')
        
        # Plot confidence ellipses
        for k in range(self.n_components):
            covs_np = self.covs[k].detach().cpu().numpy()
            v, w = np.linalg.eigh(covs_np)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = plt.matplotlib.patches.Ellipse(
                means_np[k], v[0], v[1], 180. + angle,
                color='black', fill=False, linewidth=2)
            plt.gca().add_patch(ell)
        
        plt.tight_layout(pad=0)
        plt.title(title)
        plt.legend()
        return fig
    
    def _plot_high_dim(self, X, labels=None, title="High-dimensional Gaussian Mixture"):
        """Plot high-dimensional data using PCA projection to 2D"""
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        means_np = self.means.detach().cpu().numpy()
        means_pca = pca.transform(means_np)
        
        set_seaborn_style()
        fig = plt.figure(figsize=(10, 8))
        
        # Plot projected data points
        if labels is None:
            labels = self.predict(X).cpu().numpy()
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
        
        # Plot projected means
        plt.scatter(means_pca[:, 0], means_pca[:, 1],
                   c='red', marker='x', s=200, linewidth=3,
                   label='Cluster Centers (PCA-projected)')
        
        plt.tight_layout(pad=0)
        plt.title(f"{title}\n(PCA projection to 2D)")
        plt.legend()
        return fig

# Example usage for both 2D and high-dimensional data
def demo_gmm():
    # 2D example
    torch.manual_seed(42)
    n_samples = 300
    X1 = torch.distributions.MultivariateNormal(
        torch.tensor([0., 0.]), 
        torch.tensor([[1., 0.5], [0.5, 1.]])
    ).sample((n_samples//2,))
    X2 = torch.distributions.MultivariateNormal(
        torch.tensor([4., 4.]), 
        torch.tensor([[1.5, -0.5], [-0.5, 1.5]])
    ).sample((n_samples//2,))
    X_2d = torch.cat([X1, X2])
    
    gmm_2d = GaussianMixture(n_components=2, init_strategy='kmeans')
    gmm_2d.fit(X_2d)
    labels_2d = gmm_2d.predict(X_2d)
    gmm_2d.visualize(X_2d, labels_2d, "2D Gaussian Mixture Example")
    
    # High-dimensional example (simulated 28x28 image data)
    n_samples_high = 200
    n_features = 784  # 28x28 image
    n_true_components = 3
    
    # Generate synthetic high-dimensional data
    X_high = []
    for _ in range(n_true_components):
        mean = torch.randn(n_features)
        cov = torch.eye(n_features) * torch.rand(1) * (1.5 - 0.5) + 0.5
        X_high.append(torch.distributions.MultivariateNormal(
            mean, cov
        ).sample((n_samples_high // n_true_components,)))
    X_high = torch.cat(X_high)
    
    gmm_high = GaussianMixture(n_components=n_true_components, init_strategy='kmeans')
    gmm_high.fit(X_high)
    labels_high = gmm_high.predict(X_high)
    gmm_high.visualize(X_high, labels_high, "High-dimensional Gaussian Mixture Example")

def fit_gmm(data):
    gmm = GaussianMixture(n_components=2, init_strategy='kmeans')
    gmm.fit(data)
    return gmm

if __name__ == "__main__":
    demo_gmm()
    
    from sfm.datasets import get_dataset
    
    trgtdist = get_dataset("mnist")
    data = trgtdist.sample(1000)
    gmm = fit_gmm(data)
    gmm.visualize(data, title="Fitted GMM on 8 Gaussians")
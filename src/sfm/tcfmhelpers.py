import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sfm.plotstyle import _cscheme

# from Flow_matching_tutorial.ipynb
def sample_conditional_pt(x0, x1, t, sigma, device):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    xt : Tensor, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0, device=device)
    # Gaussian interpolation?
    return mu_t + sigma * epsilon

# from Flow_matching_tutorial.ipynb
def compute_conditional_vector_field(x0, x1, device):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0

# runner/src/models/components/augmentation.py

def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.0
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[
            0
        ][:, i]
    return trJ


# examples/2D_tutorials/model-comparison-plotting.ipynb
class CNF(torch.nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(
                True
            )  # first dimension reserved to divergence propagation
            # the neural network will handle the data-dynamics here
            x_out = self.net(
                torch.cat([x_in, t * torch.ones(x.shape[0], 1).type_as(x_in)], dim=-1)
            )
            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return (
            torch.cat([-trJ[:, None], x_out], 1) + 0 * x
        )  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph


def plot_trajectories(traj, n=2000, show=False, prior=True):
    """Plot trajectories of some selected samples."""
    # adapted from conditional-flow-matching/torchcfm/utils.py
    # n: max points to plot
    
    # convert to numpy
    if isinstance(traj, torch.Tensor):
        traj = traj.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(6, 6))
    if len(traj.shape) == 2:
        # there is no flow
        if prior:
            plt.scatter(traj[:n, 0], traj[:n, 1], s=10, alpha=0.8, c=_cscheme["prior"])
            plt.legend(["Prior sample z(S)"])
        else:
            plt.scatter(traj[:n, 0], traj[:n, 1], s=4, alpha=1, c=_cscheme["final"])
            plt.legend(["z(0)"])
    else:
        plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c=_cscheme["flow"])
        plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c=_cscheme["prior"])
        plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c=_cscheme["final"])
        plt.legend(["Flow", "Prior sample z(S)", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=0.0)
    if show:
        plt.show()
    return fig
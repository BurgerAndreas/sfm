import math
import torch
import torch.nn as nn

import sklearn
from torch import Tensor
from typing import *
from zuko.utils import odeint as zuko_odeint
from torchdyn.core import NeuralODE

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)


# Simple MLP architecture for the neural network modeling the vector field
class MLPwithTimeEmbedding(nn.Sequential):
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        freqs: int = 3,
        hidden_features: List[int] = [64, 64, 64],
        time_varying: bool = True,
        device: str = "cpu",
    ):
        # "positional encoding" or "time embedding" and allows the network to adjust its behavior with respect to t
        # with more granularity than by simply giving it as input the time t.
        # part of the module's state but not a parameter

        in_features += 2 * freqs

        layers = []
        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])
        super().__init__(*layers[:-1])

        self.register_buffer("freqs", torch.arange(1, freqs + 1, device=device) * torch.pi)

    def forward(self, t: Tensor, x: Tensor = None, *args, **kwargs) -> Tensor:
        # Encode time with sinusoidal features to capture periodicity
        # t: [B] -> [B, 1]
        t = self.freqs * t[..., None].to(self.freqs.device)  # [B, f]
        t = torch.cat((t.cos(), t.sin()), dim=-1)  # [B, 2f]
        t = t.expand(*x.shape[:-1], -1)  # [B, 2f]
        x = torch.cat((t, x), dim=-1)  # [B, D+2f]
        return super().forward(x)


# Continuous Normalizing Flow (ContNormFlow) class
# This class models the vector field v(t, x) that generates the probability path
# Eq. (1) in the paper defines the vector field
class ContNormFlow(nn.Module):
    def __init__(self, model: nn.Module, fmtime=False, **kwargs):
        """Continuous Normalizing Flow (ContNormFlow) class

        Our continuous normalizing flow (ContNormFlow) is a simple multi-layer perceptron (MLP).
        To learn images instead, we could use the Unet2DModel from Hugging Face's diffusers package

        Args:
            features (int): The dimensionality of the data.
            freqs (int, optional): The number of frequency components for sinusoidal encoding. Defaults to 3.
            **kwargs: Additional keyword arguments for the MLP.
        """
        super().__init__()
        self.net = model
        # diffusion / Francois time convention
        tnoise = 1.0
        tdata = 0.0
        # flow matching / TorchCFM time convention
        if fmtime:
            tdata = 1.0
            tnoise = 0.0
        # register buffers to ensure values are moved to the correct device
        self.register_buffer("tdata", torch.tensor(tdata))
        self.register_buffer("tnoise", torch.tensor(tnoise))
    
    def set_fmtime(self, fmtime: bool):
        if fmtime:
            self.tdata = torch.tensor(1.0)
            self.tnoise = torch.tensor(0.0)
        else:
            self.tdata = torch.tensor(0.0)
            self.tnoise = torch.tensor(1.0)

    def forward(self, t: Tensor, x: Tensor, y: Tensor = None) -> Tensor:
        # Forward pass through the MLP network to model the vector field
        return self.net(t=t, x=x, y=y)

    # Encode (from data to noise)
    def encode(self, x: Tensor) -> Tensor:
        return zuko_odeint(f=self, x=x, t0=self.tdata, t1=self.tnoise, phi=self.parameters())

    def decode(self, sources: Tensor) -> Tensor:
        """Go from noise to data.

        Args:
            sources (Tensor): The initial state (noise) [B, D]

        Returns:
            Tensor: generated sample [B, D]
        """
        return zuko_odeint(f=self, x=sources, t0=self.tnoise, t1=self.tdata, phi=self.parameters())

    def log_prob(self, targets: Tensor, sourcedist) -> Tensor:
        """
        Compute log-probability of data points.
        Reverse the flow from data to noise,
        then compute the log-probability of the noise under the source distribution.
        Computes the log-determinant Jacobian term as in Eq. (27)

        To compute p1(x1) we first solve the ODE in equation 31
        with initial conditions in equation 32,
        and then compute equation 33
        in Lipman et al. Flow Matching for Generative Modeling

        targets=x1: [B, D]
        return: [B]
        """
        if not hasattr(sourcedist, "log_prob"):
            # Source distribution does not have a log-probability function
            return None
        # assert targets.shape == sourcedist.log_prob(targets).shape

        I = torch.eye(targets.shape[-1], dtype=targets.dtype, device=targets.device)
        I = I.expand(*targets.shape, targets.shape[-1]).movedim(-1, 0)

        # Augmented function to compute both the derivative and trace term
        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            # Compute Jacobian of the vector field
            jacobian = torch.autograd.grad(dx, x, I, create_graph=True, is_grads_batched=True)[0]
            trace = torch.einsum("i...i", jacobian)

            # To control step size of the ODE solver, we scale the trace term
            # This adjustment is mentioned in the user comment
            # Adaptive ODE solvers choose their step size according to an estimation of the integration error.
            # For the trace-augmented ODE, odeint over estimates the integration error because the trace has large(r) absolute values, which leads to small step sizes.
            # To mitigate this without significant loss of accuracy, we multiply the trace by a factor of 10^-2
            return dx, trace * 1e-2

        # Initialize log-determinant Jacobian term
        ladj = torch.zeros_like(targets[..., 0], device=targets.device)  # [B]
        # Solve the ODE for the augmented system with trace regularization
        # Computing the log-likelihood of a ContNormFlow requires to integrate an ODE.
        # I use the odeint function provided by Zuko to do so.
        # It implements the adaptive checkpoint adjoint (ACA) method which allows for more accurate back-propagation than the standard adjoint method implemented by torchdiffeq.
        # [B, D] -> [B, D], [B]
        z, ladj = zuko_odeint(f=augmented, x=(targets, ladj), t0=self.tdata, t1=self.tnoise, phi=self.parameters())

        # Final log-probability calculation with adjusted trace (scale back by 1e2)
        # log_prob: [B,2] -> [B]
        return sourcedist.log_prob(z) + ladj * 1e2


class NeuralODEWrapper(NeuralODE):
    def __init__(self, model, fmtime=True, **kwargs):
        # Without torchdyn_wrapper, the model is not compatible with torchdyn
        # Your vector field does not have `nn.Parameters` to optimize.
        super().__init__(torchdyn_wrapper(model), **kwargs)
        # super().__init__(model, return_t_eval=False, **kwargs)

        # diffusion / Francois time convention
        tnoise = 1.0
        tdata = 0.0
        # flow matching / TorchCFM time convention
        if fmtime:
            tdata = 1.0
            tnoise = 0.0
        # register buffers to ensure values are moved to the correct device
        self.register_buffer("tdata", torch.tensor(tdata))
        self.register_buffer("tnoise", torch.tensor(tnoise))
    
    def set_fmtime(self, fmtime: bool):
        if fmtime:
            self.tdata = 1.0
            self.tnoise = 0.0
        else:
            self.tdata = 0.0
            self.tnoise = 1.0

    def decode(self, sources: Tensor) -> Tensor:
        # [T, B, D]
        traj = self.trajectory(
            x=sources,
            t_span=torch.linspace(0, 1, 100, device=self.device),
        )
        return traj[-1]

    def forward(self, t, x, y=None, *args, **kwargs):
        # swap around x and t
        return super().forward(x, t, y, *args, **kwargs)

    # TODO: doesn't work yet
    def log_prob(self, targets: Tensor, sourcedist) -> Tensor:
        if sourcedist.log_prob(torch.tensor(1.0)) is None:
            # Source distribution does not have a log-probability function
            return None
        # assert targets.shape == sourcedist.log_prob(targets).shape

        I = torch.eye(targets.shape[-1], dtype=targets.dtype, device=targets.device)
        I = I.expand(*targets.shape, targets.shape[-1]).movedim(-1, 0)

        # Augmented function to compute both the derivative and trace term
        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            # Compute Jacobian of the vector field
            jacobian = torch.autograd.grad(dx, x, I, create_graph=True, is_grads_batched=True)[0]
            trace = torch.einsum("i...i", jacobian)

            # To control step size of the ODE solver, we scale the trace term
            # This adjustment is mentioned in the user comment
            # Adaptive ODE solvers choose their step size according to an estimation of the integration error.
            # For the trace-augmented ODE, odeint over estimates the integration error because the trace has large(r) absolute values, which leads to small step sizes.
            # To mitigate this without significant loss of accuracy, we multiply the trace by a factor of 10^-2
            return dx, trace * 1e-2

        # Initialize log-determinant Jacobian term
        ladj = torch.zeros_like(targets[..., 0])  # [B]
        # Solve the ODE for the augmented system with trace regularization
        # Computing the log-likelihood of a ContNormFlow requires to integrate an ODE.
        # I use the odeint function provided by Zuko to do so.
        # It implements the adaptive checkpoint adjoint (ACA) method which allows for more accurate back-propagation than the standard adjoint method implemented by torchdiffeq.
        # [B, D] -> [B, D], [B]
        # z, ladj = odeint(f=augmented, x=(targets, ladj), t0=0.0, t1=1.0, phi=self.parameters())
        node = NeuralODEWrapper(augmented)
        traj = node.trajectory(
            x=targets,
            t_span=torch.linspace(start=self.tdata, end=self.tnoise, steps=100, device=self.device),
        )
        z, ladj = traj[-1]

        # Final log-probability calculation with adjusted trace (scale back by 1e2)
        # log_prob: [B,2] -> [B]
        return sourcedist.log_prob(z) + ladj * 1e2


class torchdyn_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format.
    Replaces ContNormFlow with torchdyn neural ode.
    from torchcfm.utils.torch_wrapper
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, y=None, *args, **kwargs):
        # t: [] -> [B, 1]
        # x: [B, D]
        # model: [B, D+1] -> [B, D]
        # TODO: torchdyn only accepts one input tensor
        return self.model(x=x, t=t.repeat(x.shape[0])[:, None], y=y)
        # return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


class LipmanFMLoss(nn.Module):
    def __init__(self, v: nn.Module, sigma: float = 1e-4):
        """CFM loss based on Equation 23 in the flow matching paper
        https://arxiv.org/pdf/2210.02747.pdf

        ψ_t(x) = y: conditional flow
        u:          vector field
        v:          ContNormFlow vector field v
        p:          probability density path pt(x)
        σmin:       small variance at true samples
        x:          data
        z:          noise (x0 in the paper, x1 in the code)
        """
        super().__init__()
        self.v = v
        self.sigma = sigma
        self.fmtime = False # t=0 is data, t=1 is noise

    def forward(self, sources: Tensor, targets: Tensor) -> Tensor:
        """Optimal Transport conditional VF of a Gaussian.
        Diffusion time convention: t=0 is data, t=1 is noise
        sources=x1: source samples
        targets=x0: data samples
        OT for Gaussian:
        Mean and std change linearly with t:
        μt(x) = t*x1
        σt(x) = 1 - (1 - σmin)*t (20)
        """
        # Sample random time step from [0, 1]
        t = torch.rand_like(targets[..., 0, None], device=targets.device)

        # Interpolation between data and noise
        # psi = ψ = μt(x) + σt(x)*x0
        psi = (1 - t) * targets + (self.sigma + (1 - self.sigma) * t) * sources
        # Target vector field u in Eq. (21) and Eq. (23)
        # loss = ||vt(ψt(x0)) − x1 − (1 − σmin)x0||
        u = (1 - self.sigma) * sources - targets
        return (self.v(t.squeeze(-1), psi) - u).square().mean()


class LipmanTCFMLoss(nn.Module):
    def __init__(self, v: nn.Module, sigma: float = 1e-1):
        """TorchCFM version of the Lipman loss"""
        super().__init__()
        self.v = v  # NeuralODE
        self.sigma = sigma
        self.fm = TargetConditionalFlowMatcher(sigma=sigma)
        self.fmtime = True # t=1 is data, t=0 is noise

    def forward(self, sources: Tensor, targets: Tensor, labels: Tensor = None) -> Tensor:
        # sources=x0: source samples
        # targets=x1: data samples
        # labels=y: optional class label for conditional generation
        # [B], [B, D], [B, D]
        t, xt, ut = self.fm.sample_location_and_conditional_flow(sources, targets)

        # [B, D+1] -> [B, D]
        # vt = model(torch.cat([xt, t[:, None]], dim=-1))
        vt = self.v(t=t, x=xt, y=labels)

        return torch.mean((vt - ut) ** 2)


class CFMLoss(LipmanTCFMLoss):
    def __init__(self, v: nn.Module, sigma: float = 1e-1):
        """TorchCFM version of non-OT loss from Tong et al."""
        super().__init__(v=v, sigma=sigma)
        self.fm = ConditionalFlowMatcher(sigma=sigma)


class OTCFMLoss(LipmanTCFMLoss):
    def __init__(self, v: nn.Module, sigma: float = 1e-1):
        """TorchCFM version of OT loss from Tong et al."""
        super().__init__(v=v, sigma=sigma)
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)

import math
import torch
import torch.nn as nn

import sklearn
from torch import Tensor
from typing import *
from zuko.utils import odeint

from sfm.distributions import get_source_distribution


# Simple MLP architecture for the neural network modeling the vector field
class MLPwithTimeEmbedding(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        freqs: int = 3,
        hidden_features: List[int] = [64, 64],
        time_varying: bool = True,
    ):
        # "positional encoding" or "time embedding" and allows the network to adjust its behavior with respect to t
        # with more granularity than by simply giving it as input the time t.
        # part of the module's state but not a parameter
        if time_varying is False:
            nfreqs = 0
        else:
            nfreqs = int(max(0, freqs))
            in_features = in_features + 2 * nfreqs
        self.nfreqs = nfreqs
        
        layers = []
        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])
        super().__init__(*layers[:-1])
        
        if self.nfreqs > 0:
            self.register_buffer("freqs", torch.arange(1, self.nfreqs + 1) * torch.pi)
    
    def forward(self, t: Tensor, x: Tensor = None) -> Tensor:
        # Encode time with sinusoidal features to capture periodicity
        if self.nfreqs > 0:
            # t: [B] -> [B, 1]        
            t = self.freqs * t[..., None] # [B, f]
            t = torch.cat((t.cos(), t.sin()), dim=-1) # [B, 2f]
            t = t.expand(*x.shape[:-1], -1) # [B, 2f]
            x = torch.cat((t, x), dim=-1)
        elif self.nfreqs == 0:
            pass
        else:
            t = t[..., None] # [B, 1]
            x = torch.cat((t, x), dim=-1)
        return super().forward(x) # [B, D+2f]

# Continuous Normalizing Flow (CNF) class
# This class models the vector field v(t, x) that generates the probability path
# Eq. (1) in the paper defines the vector field
class CNF(nn.Module):
    def __init__(self, model: nn.Module, source: Dict, **kwargs):
        """Continuous Normalizing Flow (CNF) class

        Our continuous normalizing flow (CNF) is a simple multi-layer perceptron (MLP).
        To learn images instead, we could use the Unet2DModel from Hugging Face's diffusers package

        Args:
            features (int): The dimensionality of the data.
            freqs (int, optional): The number of frequency components for sinusoidal encoding. Defaults to 3.
            source (str, optional): The source distribution for the flow. Defaults to "gaussian".
            **kwargs: Additional keyword arguments for the MLP.
        """
        super().__init__()

        self.source = get_source_distribution(**source)
        self.net = model

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # Forward pass through the MLP network to model the vector field
        return self.net(t, x)

    # Encode (from data to latent space)
    # Uses the ODE solving strategy described in Eq. (1)
    # The solver traces the flow from data (t=0) to noise (t=1)
    def encode(self, x: Tensor) -> Tensor:
        return odeint(f=self, x=x, t0=0.0, t1=1.0, phi=self.parameters())

    # Decode (from latent space to data)
    # Reverse the flow from noise (t=1) to data (t=0)
    def decode(self, z: Tensor) -> Tensor:
        """Go from source to data.

        Args:
            z (Tensor): The initial state (noise) [B, D]

        Returns:
            Tensor: generated sample [B, D]
        """
        return odeint(f=self, x=z, t0=1.0, t1=0.0, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Compute log-probability of data points
        This computes the log-determinant Jacobian term as in Eq. (27)

        compute p1(x1) we first solve the ODE in equation 31 with initial conditions in equation 32, and the compute equation 33

        x: [B, D]
        return: [B]
        """
        if self.source.log_prob(torch.tensor(1.0)) is None:
            # Source distribution does not have a log-probability function
            return None
        # assert x.shape == self.source.log_prob(x).shape

        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        I = I.expand(*x.shape, x.shape[-1]).movedim(-1, 0)

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
        ladj = torch.zeros_like(x[..., 0])  # [B]
        # Solve the ODE for the augmented system with trace regularization
        # Computing the log-likelihood of a CNF requires to integrate an ODE.
        # I use the odeint function provided by Zuko to do so.
        # It implements the adaptive checkpoint adjoint (ACA) method which allows for more accurate back-propagation than the standard adjoint method implemented by torchdiffeq.
        # [B, D] -> [B, D], [B]
        z, ladj = odeint(f=augmented, x=(x, ladj), t0=0.0, t1=1.0, phi=self.parameters())

        # Final log-probability calculation with adjusted trace (scale back by 1e2)
        # log_prob: [B,2] -> [B]
        return self.source.log_prob(z) + ladj * 1e2



class TargetConditionalFlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        """CFM loss based on Equation 23 in the flow matching paper 
        https://arxiv.org/pdf/2210.02747.pdf

        ψ_t(x) = y: conditional flow
        u:          vector field
        v:          CNF vector field v
        p:          probability density path pt(x)
        σmin:       small variance at true samples
        x:          data
        z:          noise (x0 in the paper, x1 in the code)
        """
        super().__init__()
        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        """Optimal Transport conditional VF of a Gaussian.
        x=x1: data
        OT for Gaussian:
        Mean and std change linearly with t:
        μt(x) = t*x1
        σt(x) = 1 - (1 - σmin)*t (20)
        """
        # Sample random time step from [0, 1]
        t = torch.rand_like(x[..., 0, None])
        # Sample from the source distribution 
        z = self.v.source.sample(x.shape[0])  

        # Interpolation between data and source
        # psi = ψ = μt(x) + σt(x)*z = (1 − (1 − σmin)t)x + tx1
        psi = (1 - (1 - 1e-4) * t) * z + (t * x)
        # Target vector field u in Eq. (21) and Eq. (23)
        # loss = ||vt(ψt(x0)) − x1 − (1 − σmin)x0||
        u = x + (1 - 1e-4) * z  
        return (self.v(t.squeeze(-1), psi) - u).square().mean()

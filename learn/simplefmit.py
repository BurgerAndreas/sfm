#!/usr/bin/env python

import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import make_moons
from torch import Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import odeint


def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

tnoise = 1.
tdata = 0.

reverse_time = False
reverse_fm = False

if reverse_time:
    tdata, tnoise = tnoise, tdata

class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])

        super().__init__(*layers[:-1])


class ContNormFlow(nn.Module):
    def __init__(self, features: int, freqs: int = 3, **kwargs):
        super().__init__()

        self.net = MLP(2 * freqs + features, features, **kwargs)

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(*x.shape[:-1], -1)

        return self.net(torch.cat((t, x), dim=-1))

    def encode(self, x: Tensor) -> Tensor:
        return odeint(self, x, tdata, tnoise, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        return odeint(self, z, tnoise, tdata, phi=self.parameters())

    def log_prob(self, x: Tensor) -> Tensor:
        I = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
        I = I.expand(*x.shape, x.shape[-1]).movedim(-1, 0)

        def augmented(t: Tensor, x: Tensor, ladj: Tensor) -> Tensor:
            with torch.enable_grad():
                x = x.requires_grad_()
                dx = self(t, x)

            jacobian = torch.autograd.grad(dx, x, I, create_graph=True, is_grads_batched=True)[0]
            trace = torch.einsum('i...i', jacobian)

            return dx, trace * 1e-2

        ladj = torch.zeros_like(x[..., 0])
        z, ladj = odeint(augmented, (x, ladj), tdata, tnoise, phi=self.parameters())

        # log_normal: [B,2] -> [B]
        return log_normal(z) + ladj * 1e2


class FlowMatchingLoss(nn.Module):
    def __init__(self, v: nn.Module):
        super().__init__()

        self.v = v

    def forward(self, x: Tensor) -> Tensor:
        t = torch.rand_like(x[..., 0, None])
        z = torch.randn_like(x)
        
        # t = (1 - t) # if reverse_time else t
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x
        
        if reverse_fm:
            y = (1 - (1 - 1e-4) * t) * z + (t * x)
            u = (1 - 1e-4) * z - x

        return (self.v(t.squeeze(-1), y) - u).square().mean()


if __name__ == '__main__':
    batch_size = 256
    flow = ContNormFlow(2, hidden_features=[64] * 3)

    # Training
    loss = FlowMatchingLoss(flow)
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)

    data, _ = make_moons(16384, noise=0.05)
    data = torch.from_numpy(data).float()

    # Eval 
    # Log-likelihood
    with torch.no_grad():
        log_p = flow.log_prob(data[:batch_size])
        print(f"Log probability: {log_p.mean():.3f} ± {log_p.std():.3f}")

    # Training
    for epoch in tqdm(range(16384), ncols=44):
        subset = torch.randint(0, len(data), (batch_size,))
        x = data[subset]

        loss(x).backward()

        optimizer.step()
        optimizer.zero_grad()

    # Sampling
    with torch.no_grad():
        z = torch.randn(16384, 2)
        x = flow.decode(z)

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    plt.figure(figsize=(4.8, 4.8), dpi=150)
    plt.hist2d(*x.T, bins=64)
    plt.savefig('moons_fm.pdf')
    print("Saved figure to moons_fm.pdf")

    # Log-likelihood
    with torch.no_grad():
        log_p = flow.log_prob(data[:batch_size])
        print(f"Log probability: {log_p.mean():.3f} ± {log_p.std():.3f}")

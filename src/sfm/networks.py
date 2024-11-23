import torch 
import torch.nn as nn
from torch import Tensor
from typing import List

from torchcfm.models.unet import UNetModel

# MLP architecture with time embedding for the neural network modeling the vector field
class MLPSepTimeEmb(nn.Sequential):
    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 2,
        freqs: int = 3,
        hidden_features: List[int] = [64, 64, 64],
        time_varying: bool = True,
        device: str = "cpu",
        **kwargs,
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
    

# Simple MLP from TorchCFM
class MLPTime(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, **kwargs):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MLPTimeEmb(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, freqs=3, device="cpu", **kwargs):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.register_buffer("freqs", torch.arange(1, freqs + 1, device=device) * torch.pi)
        self.net = torch.nn.Sequential(
            # multiply by 2 because we have cos and sin
            torch.nn.Linear(dim + 2 * freqs, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        # Encode time with sinusoidal features to capture periodicity
        # The last dimension of x is the time t
        t = self.freqs * x[..., -1:] * torch.pi  # [B, D]
        t = torch.cat((t.cos(), t.sin()), dim=-1)  # [B, 2D]
        x = torch.cat((t, x[..., :-1]), dim=-1)  # [B, D+2D]
        return self.net(x)


class UNetWrapper(UNetModel):
    def __init__(self, *args, **kwargs):
        # only pass the args that are needed
        super().__init__(
            dim=kwargs["dim"],
            num_channels=kwargs["num_channels"],
            num_res_blocks=kwargs["num_res_blocks"],
            class_cond=kwargs["class_cond"],
            num_classes=kwargs["num_classes"],
        )
    
    def forward(self, t, x, y=None):
        return super().forward(t, x, y)

_models = {
    "mlp": MLPTime,
    "mlptime": MLPTimeEmb,
    "unet": UNetWrapper,
}

def get_model(trgt: str, **kwargs):
    return _models[trgt](**kwargs)


if __name__ == "__main__":
    
    # example: conditional-flow-matching/examples/images/conditional_mnist.ipynb
    # model = UNetModel(
    #     dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
    # )
    model = get_model(
        "unet", 
        dim=(1, 28, 28), 
        num_channels=32, 
        num_res_blocks=1, 
        num_classes=10, 
        class_cond=True
    )

import torch
import math

from torch import Tensor
from src.sfm.distributions import StandardNormalSource

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

def test_log_prob():
    source = StandardNormalSource(data_dim=2)
    x = torch.randn(100, 2)
    assert torch.allclose(source.log_prob(x), log_normal(x))

if __name__ == "__main__":
    test_log_prob()
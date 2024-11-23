# toy datasets
# sklearn and torchdyn both have moons and spirals
# but they are slightly different

import torch
import torchdyn.datasets as tdyndata
from sklearn.datasets import make_moons, make_swiss_roll

from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

def tdyn_moons_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_moons(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -1.00) / (3.00)
    data[:, 1] = (data[:, 1] - -0.50) / (1.50)
    return data


def tdyn_spirals_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_spirals(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -12.61) / (25.21)
    data[:, 1] = (data[:, 1] - -11.78) / (23.57)
    return data


def tdyn_gaussians_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_gaussians(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -0.94) / (1.84)
    data[:, 1] = (data[:, 1] - -0.87) / (1.73)
    return data


def tdyn_gaussians_spiral_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_gaussians_spiral(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -1.90) / (3.27)
    data[:, 1] = (data[:, 1] - -1.59) / (3.82)
    return data


def tdyn_diffeqml_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_diffeqml(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -2.02) / (4.05)
    data[:, 1] = (data[:, 1] - -2.02) / (4.04)
    return data


def tdyn_concentric_spheres_nrmed01(*args, **kwargs):
    data, _ = tdyndata.generate_concentric_spheres(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -0.99) / (1.98)
    data[:, 1] = (data[:, 1] - -0.92) / (1.90)
    data[:, 2] = (data[:, 2] - -0.99) / (1.88)
    return data


########################################################################
# sklearn

def make_moons_nrmed01(*args, **kwargs) -> torch.Tensor:
    data, _ = make_moons(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -1.00) / (3.00)
    data[:, 1] = (data[:, 1] - -0.50) / (1.50)
    return torch.tensor(data)


def make_swiss_roll_nrmed01(*args, **kwargs) -> torch.Tensor:
    data, _ = make_swiss_roll(*args, **kwargs)
    data[:, 0] = (data[:, 0] - -9.48) / (22.07)
    data[:, 1] = (data[:, 1] - 0.10) / (20.83)
    data[:, 2] = (data[:, 2] - -11.04) / (25.09)
    return torch.tensor(data)

########################################################################
# helper functions

def print_normed_func(func, *args, **kwargs):
    data, _ = func(*args, **kwargs)
    print("")
    print(f"def {func.__name__}_nrmed01(*args, **kwargs):")
    print(f"    data, _ = {func.__name__}(*args, **kwargs)")
    print(f"    data[:, 0] = (data[:, 0] - {data[:, 0].min():.2f}) / ({data[:, 0].max() - data[:, 0].min():.2f})")
    print(f"    data[:, 1] = (data[:, 1] - {data[:, 1].min():.2f}) / ({data[:, 1].max() - data[:, 1].min():.2f})")
    if data.shape[1] == 3:
        print(f"    data[:, 2] = (data[:, 2] - {data[:, 2].min():.2f}) / ({data[:, 2].max() - data[:, 2].min():.2f})")
    print(f"    return data")
    print("")

########################################################################
# MNIST

class MNISTWrapper:
    def __init__(self, **kwargs):
        self.dmin = kwargs.get("dmin", 0)
        self.dmax = kwargs.get("dmax", 1)
        
        self.trainset = datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # normalize to [-1, 1]
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        
    def sample(self, n_samples):
        # Get random indices
        indices = torch.randperm(len(self.trainset))[:n_samples]
        
        # Get samples and flatten
        samples = torch.stack([self.trainset[i][0] for i in indices])
        samples = samples.view(n_samples, -1)
        
        # Scale to [dmin, dmax] range
        # samples = samples * (self.dmax - self.dmin) + self.dmin
        return samples

########################################################################

_datasets = {
    "moons": tdyn_moons_nrmed01,
    "spirals": tdyn_spirals_nrmed01,
    "gaussians": tdyn_gaussians_nrmed01,
    "gaussians_spiral": tdyn_gaussians_spiral_nrmed01,
    "concentric_spheres": tdyn_concentric_spheres_nrmed01, # 3D
    "diffeqml": tdyn_diffeqml_nrmed01,
    # sklearn
    "skmoons": make_moons_nrmed01,
    "skspirals": make_swiss_roll_nrmed01, # 3D
}

def sample_dataset(trgt: str, *args, **kwargs) -> torch.Tensor:
    return _datasets[trgt](*args, **kwargs)

class DatasetDummy:
    def __init__(self, trgt: str, **kwargs):
        self.dmin = kwargs.get("dmin", 0)
        self.dmax = kwargs.get("dmax", 1)
        self.samplefct = lambda n_samples: sample_dataset(trgt, n_samples=n_samples, **kwargs)
        
    def sample(self, n_samples):
        samples = self.samplefct(n_samples)
        # samples are already in [0, 1]
        # samples = (samples - samples.min()) / (samples.max() - samples.min())
        samples = samples * (self.dmax - self.dmin) + self.dmin
        return samples

def get_dataset(trgt: str, *args, **kwargs):
    if trgt == "mnist":
        return MNISTWrapper(**kwargs)
    else:
        return DatasetDummy(trgt, *args, **kwargs)

if __name__ == "__main__":
    # print new function for normalized data
    print_normed_func(tdyndata.generate_moons)
    print_normed_func(tdyndata.generate_spirals)
    print_normed_func(tdyndata.generate_gaussians)
    print_normed_func(tdyndata.generate_gaussians_spiral)
    print_normed_func(tdyndata.generate_diffeqml)
    print_normed_func(tdyndata.generate_concentric_spheres)
    
    print_normed_func(make_moons)
    print_normed_func(make_swiss_roll)

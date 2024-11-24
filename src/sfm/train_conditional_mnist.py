import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel

from sfm.loggingwrapper import LoggingWrapper


# for node.trajcetory
# SOLVER_DICT = {
#     'euler': Euler, 'midpoint': Midpoint,
#     'rk4': RungeKutta4, 'rk-4': RungeKutta4, 'RungeKutta4': RungeKutta4,
#     'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
#     'tsit5': Tsitouras45, 'Tsitouras45': Tsitouras45, 'Tsitouras5': Tsitouras45,
#     'ieuler': ImplicitEuler, 'implicit_euler': ImplicitEuler,
#     'alf': AsynchronousLeapfrog, 'AsynchronousLeapfrog': AsynchronousLeapfrog
# }

def train_conditional_mnist(args: DictConfig) -> None:
    savedir = "models/cond_mnist"
    os.makedirs(savedir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 128
    n_epochs = 10
    USE_OT = False
    USE_TORCH_DIFFEQ = True # use torchdiffeq for ODE solving
    USE_WANDB = False
    
    d_img = (1, 28, 28)

    runname = "cfm-mnist"
    if USE_OT:
        runname += "-ot"

    trainset = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    # Fit a Gaussian to the training data
    all_data = torch.stack([data[0] for data in train_loader], dim=0).view(-1, 28*28)
    mean = all_data.mean(dim=0) # [784]
    std = all_data.std(dim=0) # [784]
    std = std.clamp(min=1e-5)  # ensure positive std values

    # Create source distribution as multivariate normal with same mean/std as data
    source_dist = torch.distributions.Normal(
        loc=mean.to(device),
        scale=std.to(device)
    )

    #################################
    # Class Conditional CFM / OT-CFM
    #################################

    sigma = 0.0
    model = UNetModel(
        dim=d_img, num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    if USE_OT:
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=sigma)
    # Target FM (Lipman et al. 2023)
    # FM = TargetConditionalFlowMatcher(sigma=sigma)
    node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    for epoch in range(n_epochs):
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            x1 = data[0].to(device) # [B, 1, 28, 28]
            y = data[1].to(device)  # class labels
            
            # sample source distribution
            x0 = torch.randn_like(x1)  # random noise # TODO@Source
            
            # Flow Matching
            if USE_OT:
                t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0, x1, y1=y)
                vt = model(t, xt, y1)
            else:
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                vt = model(t, xt, y)
            loss = torch.nanmean((vt - ut) ** 2)
            # Backprop
            loss.backward()
            optimizer.step()
            tqdm.write(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.3}")
    
    # save model
    torch.save(model.state_dict(), f"{savedir}/{runname}.pth")
    
    sqrt_eval_batch_size = 10
    evalbs = sqrt_eval_batch_size**2
    
    # generate samples 
    generated_class_list = torch.arange(10, device=device).repeat(sqrt_eval_batch_size // 10 + 1) # [100]
    generated_class_list = generated_class_list[:sqrt_eval_batch_size]
    with torch.no_grad():
        y0 = source_dist.sample(evalbs).to(device)
        # reshape to [evalbs, *d_img]
        y0 = y0.view(evalbs, *d_img)
        if USE_TORCH_DIFFEQ:
            traj = torchdiffeq.odeint(
                func=lambda t, x: model.forward(t, x, generated_class_list),
                y0=y0,
                # t is not the integration steps but the time points for which to solve for
                # The first element of is taken to be the initial time point
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5", # default solver
                # for a fixed step solver, we need to specify the step size
                # method="euler", # euler, midpoint, rk4, heun3
                # step_size=0.01,
            )
        else:
            traj = node.trajectory(
                x=y0,
                t_span=torch.linspace(0, 1, 2, device=device),
            )
    grid = make_grid(
        traj[-1, :evalbs].view([-1, *d_img]).clip(-1, 1), 
        value_range=(-1, 1), padding=0, nrow=sqrt_eval_batch_size
    )
    img = ToPILImage()(grid)
    plt.imshow(img)
    plt.savefig(f"{savedir}/traj_{runname}.png")
    print(f"Saved trajectory to {savedir}/traj_{runname}.png")
    plt.close()
    
    print("Done!")

@hydra.main(config_name="tcfm", config_path="./config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    train_conditional_mnist(args)
    
if __name__ == "__main__":
    hydra_wrapper()
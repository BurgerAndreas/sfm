# Flow Matching with source distributions

## Pytorch

```bash
mamba env remove --name sfm -y
mamba create -n sfm python=3.11 -y
mamba activate sfm

mamba install numpy==1.24.*3* matplotlib tqdm scikit-learn==1.3.* zuko omegaconf hydra-core wandb neptune black -y
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch
pip install torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/cu121/torchvision
```

based on [Francois Rozet's gist](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa)
and [TorchCFM](https://github.com/atong01/conditional-flow-matching)



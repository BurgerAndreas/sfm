# Flow Matching with source distributions

based on [Francois Rozet's gist](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa)
and [TorchCFM](https://github.com/atong01/conditional-flow-matching)


## Run

```bash
mamba activate sfm
python scripts/run_flow_matching.py
```
We use hydra to manage the config.

```bash
python scripts/run_flow_matching.py fmloss=lipmantcfm fmtime=true
# these are the most interesting, since they work with arbitrary source distributions
python scripts/run_flow_matching.py fmloss=cfm fmtime=true
python scripts/run_flow_matching.py fmloss=otcfm fmtime=true
```

To plot trajectories
```bash
python scripts/run_flow_matching.py fmloss=cfm fmtime=true n_ode=torchdyn
```

## Installation

```bash
mamba env remove --name sfm -y
mamba create -n sfm python=3.11 -y
mamba activate sfm

mamba install requirements.txt -y
pip install torchcfm==0.1.0

# install torch however needed for your system
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch
pip install torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/cu121/torchvision
```




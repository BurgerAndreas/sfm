# Flow Matching beyond Gaussian Source Distributions

Based on [Francois Rozet's gist](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa)
and [TorchCFM](https://github.com/atong01/conditional-flow-matching)


## Run TorchCFM

```bash
mamba activate sfm
python scripts/tcfm.py
```

## Run other flow matching losses

```bash
mamba activate sfm
```
We use hydra to manage configs. To compare different flow matching losses:
```bash
python scripts/run_flow_matching.py fmloss=lipman
python scripts/run_flow_matching.py fmloss=lipmantcfm
# these are the most interesting, since they work with arbitrary source distributions
python scripts/run_flow_matching.py fmloss=cfm
python scripts/run_flow_matching.py fmloss=otcfm
```

To plot trajectories
```bash
python scripts/run_flow_matching.py fmloss=cfm n_ode=torchdyn
```

Try different source distributions:
```bash
# python scripts/run_flow_matching.py fmloss=cfm source=normal logger=neptune

sources=("normal" "uniform" "beta" "laplace" "mog" "cauchy" "fisher" "studentt" "weibull" "gamma" "laplace" "gumbel")
losses=("cfm" "otcfm")
for source in "${sources[@]}"; do
    for loss in "${losses[@]}"; do
        python scripts/run_flow_matching.py fmloss=${loss} source=${source} logger=neptune tags=["s1"]
    done
done
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

## TODO

[] start from gaussian source
[] two gaussians source / mog source

- [] Measure Wasserstein distance between source and target
- [] Select runs for loss/log_prob plots



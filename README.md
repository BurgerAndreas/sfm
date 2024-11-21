# Flow Matching beyond Gaussian Source Distributions

Based on [Francois Rozet's gist](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa)
and [ Alexander Tong and Kilian Fatras's TorchCFM](https://github.com/atong01/conditional-flow-matching) 🙏


## Run TorchCFM

```bash
mamba activate sfm

sources=("8gaussians" "gamma" "laplace" "gaussian" "normal" "uniform" "mog")
ots=(True False)

for use_ot in "${ots[@]}"; do
    for source in "${sources[@]}"; do
        python scripts/tcfm.py source=${source} task=all use_ot=${use_ot}
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

## Run other flow matching losses (experimental)

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

## TODO


- [] Measure Wasserstein distance between source and target



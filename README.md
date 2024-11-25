# Flow Matching beyond Gaussian Source Distributions

Based on [Francois Rozet's gist](https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa)
and [Alexander Tong and Kilian Fatras's TorchCFM](https://github.com/atong01/conditional-flow-matching) 🙏


## Run TorchCFM

```bash
mamba activate sfm

# Make moons
sources=("8gaussians" "gamma" "beta" "cauchy" "diagonal" "laplace" "gaussian" "normal" "uniform" "mog" "multivariate" "datafittednormal")
ots=(True False)

for use_ot in "${ots[@]}"; do
    for source in "${sources[@]}"; do
        python scripts/tcfm.py source=${source} task=all use_ot=${use_ot} 
    done
done

# MNIST
sources=("gamma" "beta" "diagonal" "laplace" "normal" "uniform" "mog" "multivariate" "datafittednormal" "8gaussians" "gaussian")
ots=(True False)
for use_ot in "${ots[@]}"; do
    for source in "${sources[@]}"; do
        python scripts/tcfm.py source=${source} use_ot=${use_ot} data=mnist
    done
done

# fit a GMM and use it as source distribution
python scripts/tcfm.py source=gmm data=mnist

# use Lipman flow matching (only for normal source)
python scripts/tcfm.py source=normal fmloss=lipman
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

- [] MNIST integrator logprob
- [] GMM on MNIST

- [] sweep training with different inference steps
- [] plot different inference steps sidebyside

- [] Measure (Wasserstein) distance between source and target



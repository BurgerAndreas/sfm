import hydra
import omegaconf
import yaml
import torch
import os
from sfm.distributions import get_source_distribution


def test_source_configs(n_samples=1000, dmin=-1, dmax=1, atol=0.2):

    # loop over src/sfm/config/source/
    currentfile = os.path.abspath(__file__)
    currentdir = os.path.dirname(currentfile)
    # basefile = os.path.abspath(f"{currentdir}/../src/sfm/config/base.yaml")
    source_dir = os.path.abspath(f"{currentdir}/../src/sfm/config/source/")

    for fname in os.listdir(source_dir):
        with open(f"{source_dir}/{fname}", "r") as f:
            cfg = omegaconf.OmegaConf.load(f)

            print(f"{fname}", end="")
            try:
                dist = get_source_distribution(**cfg["source"])
            except Exception as e:
                print(f"\n cfg: \n{cfg['source']}")
                raise e

            samples = dist.sample(n_samples)
            assert samples.shape == (n_samples, 2), f"\n Samples of {fname} have wrong shape {samples.shape}"

            # samples = torch.randn(n_samples, 2)
            log_prob = dist.log_prob(samples)
            assert log_prob.shape == (n_samples,), f"\n Log-probability of {fname} has wrong shape {log_prob.shape}"
            print(" ✅")

            minx = samples[:, 0].min()
            maxx = samples[:, 0].max()
            miny = samples[:, 1].min()
            maxy = samples[:, 1].max()
            print(f" minx={minx:.2f}, maxx={maxx:.2f}, miny={miny:.2f}, maxy={maxy:.2f}", end="")
            # if within +-atol% of dmin or dmax, then it's ok
            if (minx >= dmin - atol) \
            and (maxx <= dmax + atol) \
            and (miny >= dmin - atol) \
            and (maxy <= dmax + atol):
                print(" ✅")
            else:
                print(" ❌")


    print("\nAll tests passed ✅")


if __name__ == "__main__":
    test_source_configs()

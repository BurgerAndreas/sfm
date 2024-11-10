import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from sfm.trainer import Trainer

"""
In score-based generative modeling, it is standard to set t=0 as the data (noiseless) extremity and t=1 as the noise extremity. 
In the Flow Matching paper the authors reverse it, but we use the score-based convention in this implementation.
"""


@hydra.main(config_name="base", config_path="../src/sfm/config", version_base="1.3")
def run_with_hydra(args: DictConfig) -> None:
    # to dict
    cfg = OmegaConf.structured(OmegaConf.to_yaml(args))

    # Training
    trainer = Trainer(cfg)
    losses, log_probs = trainer.train()

    # Final evaluation
    gensamples, log_p = trainer.evaluate()
    trainer.logger.log({"log_p": log_p.nanmean()}, step=trainer.step, split="val")
    log_probs.append(log_p.nanmean().item())

    # save log_probs and losses to file
    trainer.savetofile(log_probs, "logprobs")
    trainer.savetofile(losses, "losses")

    # Plot the generated samples and the loss
    fname, fig = trainer.plot_data(x=gensamples, step=trainer.step)
    print(f"Saved data plot to\n {fname}")
    fname, fig = trainer.plot_loss(losses=losses)
    print(f"Saved loss plot to\n {fname}")
    fname, fig = trainer.plot_loss(losses=log_probs, fname="logprobs")
    print(f"Saved logprobs plot to\n {fname}")

    # Plot the training and validation distributions
    fname, fig = trainer.plot_data(x=trainer.data_train, step=0, folder="plots/data", fname=f"{cfg['data']['trgt']}_train.png")
    print(f"Saved training data plot to\n {fname}")
    fname, fig = trainer.plot_data(x=trainer.data_val, step=0, folder="plots/data", fname=f"{cfg['data']['trgt']}_val.png")
    print(f"Saved validation data plot to\n {fname}")

    print(
        f"Log probability: {log_p.nanmean():.2f} ± {torch.std(log_p[~log_p.isnan()]):.2f}" 
        f"(default: -0.5 ± 0.7)"
    )
    print(f"Loss after {cfg['n_samples']} n_samples: {losses[-1]:.3f}")

    trainer.finalize()
    print("\nDone ✅")


if __name__ == "__main__":
    run_with_hydra()

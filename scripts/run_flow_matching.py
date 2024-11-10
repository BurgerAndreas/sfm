import hydra
from omegaconf import DictConfig, OmegaConf

from sfm.trainer import Trainer, plot_data, plot_loss

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
    losses = trainer.train()

    # Final evaluation
    gensamples, log_p = trainer.evaluate()
    trainer.logger.log({"log_p": log_p.mean()}, step=trainer.step, split="val")

    # Plot the generated samples and the loss
    plot_data(x=gensamples, cfg=cfg, step=trainer.step)
    plot_loss(losses=losses, cfg=cfg)

    # Plot the training and validation distributions
    plot_data(x=trainer.data_train, cfg=cfg, step=0, folder="plots/data", fname=f"{cfg['dataset']}_train.png")
    plot_data(x=trainer.data_val, cfg=cfg, step=0, folder="plots/data", fname=f"{cfg['dataset']}_val.png")

    print(f"Log probability: {log_p.mean():.2f} ± {log_p.std():.2f} (default: -0.5 ± 0.7)")
    print(f"Loss after {cfg['n_samples']} n_samples: {losses[-1]:.3f}")

    trainer.finalize()
    print("\nDone ✅")

if __name__ == "__main__":
    run_with_hydra()

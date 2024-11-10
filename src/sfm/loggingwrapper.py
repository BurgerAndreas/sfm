import neptune
import wandb
import os
import omegaconf
from omegaconf import OmegaConf
import numpy as np
import torch


class LoggingWrapper:
    def __init__(self, args, runname):
        self.run = None

    def log(self, metrics: dict, step: int, split: str):
        pass

    def log_img(self, img, step, split):
        pass

    def stop(self):
        pass


class WandbWrapper(LoggingWrapper):
    def __init__(self, args, runname):
        wandb.require("core")
        self.run = wandb.init(
            project="fm-source",
            name=runname,
            config=OmegaConf.to_container(args, resolve=True),
        )

        # if args.wandb == False:
        #     # wandb.init(mode="disabled")
        #     os.environ["WANDB_DISABLED"] = "true"

    def log(self, metrics: dict, step: int, split: str):
        metrics = {split + "/" + k: v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_img(self, img, step, split):
        wandb.log({split: wandb.Image(img)}, step=step)

    def stop(self):
        wandb.finish()


class NeptuneWrapper(LoggingWrapper):
    def __init__(self, args, runname):
        args_dict = OmegaConf.to_container(args, resolve=True)
        self.run = neptune.init_run(
            project="burgerandreas/fm-source",
            name=runname,
            capture_stderr=True,
            capture_stdout=True,
            tags=args_dict["tags"],
            # description="First BERT run for NLP project",
            # mode="async",
            # tags=["huggingface", "test", "BERT"],
            # source_files=["training_with_bert.py", "net.py"],
            # capture_hardware_metrics=False,
            # dependencies="infer",
            # # with_id="CLS-123", # resume a run
        )
        # args to dict
        # stringify_unsupported() to convert values of unsupported types to strings
        args_dict = neptune.utils.stringify_unsupported(args_dict)
        self.run["parameters"] = args_dict

    def log_safely(self, metrics: dict, step: int, split: str):
        for key, value in metrics.items():
            if "log_p" in key:
                print(f"=== log_p: {value}", type(value))
            # ValueError: Out of range float values are not JSON compliant
            if (isinstance(value, float) and np.isnan(value)) or isinstance(value, torch.Tensor) and torch.isnan(value):
                value = "NaN"
            elif (
                isinstance(value, float)
                and value == float("inf")
                or isinstance(value, torch.Tensor)
                and value == float("inf")
            ):
                value = "Infinity"
            elif (
                isinstance(value, float)
                and value == float("-inf")
                or isinstance(value, torch.Tensor)
                and value == float("-inf")
            ):
                value = "-Infinity"
            self.run[split + "/" + key].append(value, step=step)

    def log(self, metrics: dict, step: int, split: str):
        self.run[split].append(metrics, step=step)

    def log_img(self, img, key, step, split, name=None, description=None):
        # self.run[split].log(neptune.types.File.as_image(img), step=step)
        # self.run[split + "/" + key].upload(img)
        # name=name, description=description
        self.run[split + "/" + key].append(img, step=step, name=name, description=description) 

    def stop(self):
        self.run.stop()


def name_from_config(args: omegaconf.DictConfig) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    IGNORE_OVERRIDES = []
    REPLACE = {}
    try:
        # model name format:
        # deq_dot_product_attention_transformer_exp_l2_md17
        # deq_graph_attention_transformer_nonlinear_l2_md17
        mname = args["wandb_run_name"]
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        # print(f'Overrides: {args.override_dirname}')
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                override = arg.replace("+", "").replace("_", "")
                override = override.replace("=", "-").replace(".", "")
                # override = override.replace("deqkwargstest", "")
                override = override.replace("deqkwargs", "").replace("model", "")
                override_names += " " + override
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    for key, value in REPLACE.items():
        _name = _name.replace(key, value)
    # done
    print(f"Name: {_name}")
    return _name


def get_logger(args):
    runname = name_from_config(args)
    if args.logger == "wandb":
        return WandbWrapper(args, runname)
    elif args.logger == "neptune":
        return NeptuneWrapper(args, runname)
    elif args.logger in ["print", "none", "None", None, "null"]:
        return LoggingWrapper(args, runname)
    else:
        raise ValueError("Invalid logger")


class ArgsDict:
    def __init__(self, args={}):
        self._args = args

    def __getattr__(self, name):
        return self._args[name]

    def __setattr__(self, name, value):
        self._args[name] = value

    def __contains__(self, name):
        return name in self._args

    def __getitem__(self, name):
        return self._args[name]

    def __setitem__(self, name, value):
        self._args[name] = value

    def __repr__(self):
        return repr(self._args)

    def __str__(self):
        return str(self._args)

    def __iter__(self):
        return iter(self._args)

    def __len__(self):
        return len(self._args)

    def __dir__(self):
        return dir(self._args)

    def __eq__(self, other):
        return self._args == other

    def __ne__(self, other):
        return self._args != other

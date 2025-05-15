from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmtuner.extras.callbacks import LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.tuner.core import get_train_args, get_infer_args, load_model_and_tokenizer
from llmtuner.tuner.mdm.workflow import run as run_mdm
from llmtuner.tuner.mdm.trainer import CustomDiffusionTrainer
# import importlib

if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None):
    model_args, diffusion_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    logger.debug(f"dataset_list: {data_args.dataset_list}")
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if finetuning_args.stage == "sft":
        from llmtuner.tuner.sft.workflow import run
        run(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "mdm":
        run_mdm(
            trainer_cls=CustomDiffusionTrainer,
            model_args=model_args,
            diffusion_args=diffusion_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
            callbacks=callbacks
        )
    else:
        raise NotImplementedError(f"Stage {finetuning_args.stage} is not supported.")


def export_model(args: Optional[Dict[str, Any]] = None, max_shard_size: Optional[str] = "10GB"):
    model_args, _, finetuning_args, _ = get_infer_args(args)
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    model.config.use_cache = True
    tokenizer.padding_side = "left"  # restore padding side
    tokenizer.init_kwargs["padding_side"] = "left"
    model.save_pretrained(model_args.export_dir, max_shard_size=max_shard_size)
    try:
        tokenizer.save_pretrained(model_args.export_dir)
    except Exception:
        logger.warning("Cannot save tokenizer, please copy the files manually.")


if __name__ == "__main__":
    run_exp()

import lightning as L
import torch


def load_module(module_cls: type[L.LightningModule], checkpoint_path: str, **kwargs) -> L.LightningModule:
    """Instantiate `module_cls(**kwargs)` and load a saved checkpoint into it,
    transparently handling checkpoints saved from a `torch.compile`-wrapped
    model.

    `train.py` compiles the model before wrapping it in a LightningModule
    whenever `trainer.compile_model` is true (the default), and `torch.compile`
    (`OptimizedModule`) stores the real submodules under `_orig_mod`, so every
    key in the saved state_dict is prefixed `model._orig_mod....` instead of
    `model....`. That prefix only needs stripping when the checkpoint is
    loaded into a *fresh, uncompiled* model in a separate process (as
    `test.py`/`extract_tokens.py` do) -- reloading into the same,
    already-compiled module within the training run itself (e.g.
    `Trainer.test(..., ckpt_path="best")`) is unaffected and needs no such
    handling.

    Unlike `LightningModule.load_from_checkpoint`, this does not restore
    hyperparameters saved alongside the checkpoint (e.g. `learning_rate`) --
    only `kwargs` and `module_cls`'s own defaults apply. Fine for the
    inference-only scripts this is used by, which never need those.
    """
    module = module_cls(**kwargs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {key.replace("._orig_mod.", "."): value for key, value in checkpoint["state_dict"].items()}
    module.load_state_dict(state_dict)
    return module

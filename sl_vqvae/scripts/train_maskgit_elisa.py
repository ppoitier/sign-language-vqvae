import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sl_vqvae.data import load_datasets_and_dataloaders
from sl_vqvae.random import set_seed
from sl_vqvae.targets.token_target import TokenTarget

from sl_vqvae.config import BERTTrainConfig as MaskGITTrainConfig 

from WP3.modelMASKGIT import MotionMaskGIT
from sl_vqvae.trainer.maskgit_module_elisa import MaskGITTrainingModule

torch.set_float32_matmul_precision('high')


def train(config: MaskGITTrainConfig) -> None:
    seed = config.seed
    seed = 'random' if seed is None else seed
    seed = set_seed(seed)

    #feed VQ-VAE tokens
    targets = {"tokens": TokenTarget(config.data.tokens_path)}
    _, dataloaders = load_datasets_and_dataloaders(
        config.data.root,
        batch_size=config.data.batch_size,
        n_workers=config.data.num_workers,
        annotated=config.data.annotated,
        train_split=config.data.train_split,
        window_size=config.data.window_size,
        window_stride=config.data.window_stride,
        max_empty_windows=config.data.max_empty_windows,
        targets=targets,
    )

    #intantiate Maylis model
    model = MotionMaskGIT(
        codebook_size=1500, 
        d_model=768,
        nhead=12,
        num_layers=6
    )
    if config.trainer.compile_model:
        model = torch.compile(model)

    #give model and parameters to my module
    module = MaskGITTrainingModule(
        model=model,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    logger = TensorBoardLogger(save_dir=config.trainer.log_dir + f'/{seed}', name=config.trainer.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.trainer.checkpoint_dir + f'/{seed}',
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        fast_dev_run=config.trainer.fast_dev_run,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=config.trainer.enable_progress_bar,
        gradient_clip_val=config.trainer.gradient_clip_val,
        overfit_batches=config.trainer.overfit_batches,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )

    if config.trainer.run_test_after_fit and not config.trainer.fast_dev_run:
        trainer.test(module, dataloaders=dataloaders["testing"], ckpt_path="best")

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a MASKGIT-style.")
    parser.add_argument("--config", required=True, help="Path to a JSON training config file.")
    args = parser.parse_args()

    config = MaskGITTrainConfig.from_json(args.config)
    train(config)


if __name__ == "__main__":
    main()
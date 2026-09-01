import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sl_vqvae.config import TrainConfig
from sl_vqvae.data import load_datasets_and_dataloaders
from sl_vqvae.nn.vqvae.factory import build_model
from sl_vqvae.scripts.results import save_results
from sl_vqvae.trainer.vqvae_module import VQVAETrainingModule
from sl_vqvae.random import set_seed

torch.set_float32_matmul_precision('high')


def train(config: TrainConfig) -> None:
    seed = config.seed
    seed = 'random' if seed is None else seed
    seed = set_seed(seed)

    _, dataloaders = load_datasets_and_dataloaders(
        config.data.root,
        batch_size=config.data.batch_size,
        n_workers=config.data.num_workers,
    )

    model = build_model(config.model)
    if config.trainer.compile_model:
        model = torch.compile(model)

    module = VQVAETrainingModule(
        model,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        num_coordinates=config.num_coordinates,
        cache_test_outputs=config.cache_test_outputs,
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
        if config.trainer.results_path:
            save_results(module.test_outputs, config.trainer.results_path + f'/{seed}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model on sign language pose sequences.")
    parser.add_argument("--config", required=True, help="Path to a JSON training config file.")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config)
    train(config)


if __name__ == "__main__":
    main()

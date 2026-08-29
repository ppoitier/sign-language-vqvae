import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from sl_vqvae.data import load_datasets_and_dataloaders
from sl_vqvae.nn.vqvae.transformer import TransformerVQVAE
from sl_vqvae.nn.vqvae.body_part_transformer import BodyPartTransformerVQVAE
from sl_vqvae.trainer.vqvae_module import VQVAETrainingModule


def launch_training():
    root = "E:/datasets/sign-language/lsfb-cont"
    datasets, dataloaders = load_datasets_and_dataloaders(root, batch_size=64)

    model = TransformerVQVAE(embedding_dim=256, n_embeddings=1000, max_length=500)

    log_dir = "D:/data/trec26/vqvae/logs"
    checkpoint_dir = "D:/data/trec26/vqvae/checkpoints"

    module = VQVAETrainingModule(model, learning_rate=3e-4)

    logger = TensorBoardLogger(save_dir=log_dir, name="vqvae")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        max_epochs=50,
        log_every_n_steps=10,
        fast_dev_run=False,
        overfit_batches=False,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )
    trainer.test(module, dataloaders=dataloaders["testing"], ckpt_path="best")


def launch_body_part_training():
    root = "E:/datasets/sign-language/lsfb-cont"
    datasets, dataloaders = load_datasets_and_dataloaders(root, batch_size=16)

    model = BodyPartTransformerVQVAE(embedding_dim=256, max_length=500)

    log_dir = "D:/data/trec26/vqvae/logs"
    checkpoint_dir = "D:/data/trec26/vqvae/checkpoints"

    module = VQVAETrainingModule(model, learning_rate=3e-4)

    logger = TensorBoardLogger(save_dir=log_dir, name="body_part_vqvae")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    trainer = L.Trainer(
        max_epochs=50,
        log_every_n_steps=10,
        fast_dev_run=False,
        overfit_batches=False,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )
    trainer.test(module, dataloaders=dataloaders["testing"], ckpt_path="best")
    save_results(module.test_outputs, "D:/data/trec26/vqvae/results/last_v2.npy")


def save_results(results, filepath: str):
    np.save(filepath, results, allow_pickle=True)


# def launch_testing(checkpoint_path: str) -> dict:
#     root = "E:/datasets/sign-language/lsfb-cont"
#     datasets, dataloaders = load_datasets_and_dataloaders(root, batch_size=16)
#
#     model = TransformerVQVAE(embedding_dim=256, n_embeddings=1000, max_length=500)
#     module = VQVAETrainingModule.load_from_checkpoint(checkpoint_path, model=model)
#
#     trainer = L.Trainer(logger=False, enable_progress_bar=False)
#     trainer.test(module, dataloaders=dataloaders["testing"])
#
#     save_results(module.test_outputs)
#
#     return module.test_outputs


if __name__ == "__main__":
    launch_body_part_training()

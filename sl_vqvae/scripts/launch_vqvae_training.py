import lightning as L

from sl_vqvae.data import load_datasets_and_dataloaders
from sl_vqvae.criterions.vqvae_loss import VQVAELoss
from sl_vqvae.nn.vqvae.body_part_graph_tokenizer import BodyPartGraphTokenizer
from sl_vqvae.trainer.vqvae_module import VQVAETrainingModule


def launch_training():
    root = "E:/datasets/sign-language/lsfb-cont"
    datasets, dataloaders = load_datasets_and_dataloaders(root, batch_size=16)

    model = BodyPartGraphTokenizer()
    criterion = VQVAELoss(reconstruction="l1", vq_loss_weight=1.0)
    module = VQVAETrainingModule(model, criterion, learning_rate=3e-4)

    trainer = L.Trainer(
        max_epochs=50,
        precision="16-mixed",
        log_every_n_steps=10,
        fast_dev_run=True,
    )
    trainer.fit(
        module,
        train_dataloaders=dataloaders["training"],
        val_dataloaders=dataloaders["validation"],
    )


if __name__ == "__main__":
    launch_training()

import argparse

import lightning as L

from sl_vqvae.config import TestConfig
from sl_vqvae.data import load_dataloader, load_dataset
from sl_vqvae.nn.vqvae.factory import build_model
from sl_vqvae.scripts.results import save_results
from sl_vqvae.trainer.vqvae_module import VQVAETrainingModule


def test(config: TestConfig) -> dict:
    dataset = load_dataset(config.data.root, annotated=config.data.annotated, split="testing")
    dataloader = load_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = build_model(config.model)
    module = VQVAETrainingModule.load_from_checkpoint(
        config.testing.checkpoint_path,
        model=model,
        num_coordinates=config.num_coordinates,
    )

    trainer = L.Trainer(logger=False, enable_progress_bar=config.testing.enable_progress_bar)
    trainer.test(module, dataloaders=dataloader)

    if config.testing.results_path:
        save_results(module.test_outputs, config.testing.results_path)

    return module.test_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained VQ-VAE checkpoint on the test split.")
    parser.add_argument("--config", required=True, help="Path to a JSON testing config file.")
    args = parser.parse_args()

    config = TestConfig.from_json(args.config)
    test(config)


if __name__ == "__main__":
    main()

import argparse

import lightning as L

from sl_vqvae.config import TestConfig
from sl_vqvae.data import load_dataloader, load_dataset
from sl_vqvae.nn.vqvae.factory import build_model
from sl_vqvae.scripts.checkpoint import load_module
from sl_vqvae.scripts.results import save_results
from sl_vqvae.trainer.vqvae_module import VQVAETrainingModule


def extract_tokens(config: TestConfig) -> dict:
    """Tokenize the whole dataset (training + validation + testing) with a
    trained VQ-VAE checkpoint and return `{window_id: {body_part: tokens}}`.

    Reuses `VQVAETrainingModule`'s test-time token caching (see
    `VQVAETrainingModule.test_step`) instead of re-implementing inference,
    running it once per split and merging the results. The reconstructed
    poses that the same caching also collects are dropped -- only the
    token ids are kept, since they're the only thing the BERT pretraining
    stage (`sl_vqvae.targets.TokenTarget`) needs.
    """
    model = build_model(config.model)
    module = load_module(
        VQVAETrainingModule,
        config.testing.checkpoint_path,
        model=model,
        num_coordinates=config.num_coordinates,
        cache_test_outputs=True,
    )

    trainer = L.Trainer(logger=False, enable_progress_bar=config.testing.enable_progress_bar)

    tokens: dict = {}
    for split in ("training", "validation", "testing"):
        dataset = load_dataset(config.data.root, annotated=config.data.annotated, split=split)
        dataloader = load_dataloader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        trainer.test(module, dataloaders=dataloader)
        for window_id, output in module.test_outputs.items():
            tokens[window_id] = {body_part: t.numpy() for body_part, t in output["tokens"].items()}

    if config.testing.results_path:
        save_results(tokens, config.testing.results_path)

    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize the full dataset with a trained VQ-VAE checkpoint and save window_id -> tokens."
    )
    parser.add_argument("--config", required=True, help="Path to a JSON testing config file.")
    args = parser.parse_args()

    config = TestConfig.from_json(args.config)
    extract_tokens(config)


if __name__ == "__main__":
    main()

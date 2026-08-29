from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, so a typo in a JSON config
    fails loudly at load time instead of silently falling back to a default.
    """

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    root: str
    batch_size: int = 16
    num_workers: int = 0
    annotated: bool = True


class TransformerModelConfig(StrictModel):
    type: Literal["transformer"] = "transformer"
    embedding_dim: int = 256
    n_embeddings: int = 1000
    max_length: int = 500
    body_parts: list[str] = ["upper_pose", "left_hand", "right_hand"]
    n_pose_landmarks: int = 23
    n_hand_landmarks: int = 21
    n_coordinates: int = 2
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pos_encoding: str = "rope"
    attn_mask_strategy: str | None = None
    reconstruction: Literal["l1", "l2"] = "l2"
    reconstruction_weights: dict[str, float] | None = None
    commitment_loss_factor: float = 0.25
    quantizer_ema_decay: float = 0.99


class BodyPartTransformerModelConfig(StrictModel):
    type: Literal["body_part_transformer"] = "body_part_transformer"
    embedding_dim: int = 256
    n_pose_embeddings: int = 500
    n_hand_embeddings: int = 1000
    max_length: int = 500
    n_pose_landmarks: int = 23
    n_hand_landmarks: int = 21
    n_coordinates: int = 2
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pos_encoding: str = "rope"
    attn_mask_strategy: str | None = None
    reconstruction: Literal["l1", "l2"] = "l2"
    reconstruction_weights: dict[str, float] | None = None
    commitment_loss_factor: float = 0.25
    quantization_loss_factor: float = 1.0
    use_quantizer_ema: bool = True
    quantizer_ema_decay: float = 0.99


ModelConfig = Annotated[
    Union[TransformerModelConfig, BodyPartTransformerModelConfig],
    Field(discriminator="type"),
]


class OptimizerConfig(StrictModel):
    learning_rate: float = 3e-4
    weight_decay: float = 0.0


class TrainerConfig(StrictModel):
    experiment_name: str
    log_dir: str
    checkpoint_dir: str
    max_epochs: int = 50
    log_every_n_steps: int = 10
    enable_progress_bar: bool = False
    fast_dev_run: bool = False
    run_test_after_fit: bool = True
    results_path: str | None = None
    compile_model: bool = True


class TrainConfig(StrictModel):
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerConfig
    num_coordinates: int = 2
    cache_test_outputs: bool = True
    seed: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.model_validate_json(Path(path).read_text())


class TestingConfig(StrictModel):
    checkpoint_path: str
    results_path: str | None = None
    enable_progress_bar: bool = False


class TestConfig(StrictModel):
    data: DataConfig
    model: ModelConfig
    testing: TestingConfig
    num_coordinates: int = 2

    @classmethod
    def from_json(cls, path: str | Path) -> "TestConfig":
        return cls.model_validate_json(Path(path).read_text())

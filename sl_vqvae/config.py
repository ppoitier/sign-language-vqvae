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
    train_split: Literal["training", "training_with_unannotated"] = "training"
    window_size: int = 500
    window_stride: int = 400
    max_empty_windows: int | None = 0


class BERTDataConfig(DataConfig):
    # Path to the `{window_id: {body_part: tokens}}` .npy file saved by
    # `sl_vqvae.scripts.extract_tokens`, loaded by `sl_vqvae.targets.TokenTarget`.
    tokens_path: str


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


class BERTModelConfig(StrictModel):
    embedding_dim: int = 256
    body_part_mapping: dict[str, str] = {
        "upper_pose": "pose",
        "left_hand": "hand",
        "right_hand": "hand",
    }
    n_embeddings: dict[str, int] = {"pose": 500, "hand": 1000}
    n_pose_landmarks: int = 23
    n_hand_landmarks: int = 21
    gcn_hidden_dim: int = 128
    gcn_layers: int = 2
    max_length: int = 500
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pos_encoding: str = "rope"
    attn_mask_strategy: str | None = None
    mask_ratio: float = 0.4


class CSLRModelConfig(StrictModel):
    embedding_dim: int = 256
    body_parts: list[str] = ["upper_pose", "left_hand", "right_hand"]
    n_pose_landmarks: int = 23
    n_hand_landmarks: int = 21
    gcn_hidden_dim: int = 128
    gcn_layers: int = 2
    max_length: int = 500
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    pos_encoding: str = "rope"
    attn_mask_strategy: str | None = None
    vocab_size: int = 1000
    pretrained_bert_checkpoint: str | None = None


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
    gradient_clip_val: float | None = 1.0
    overfit_batches: float = 0.0
    # Epochs with no improvement on the early-stopping monitor before training
    # stops early. `None` disables early stopping.
    early_stopping_patience: int | None = 20


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


class BERTTrainConfig(StrictModel):
    data: BERTDataConfig
    model: BERTModelConfig = BERTModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerConfig
    seed: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "BERTTrainConfig":
        return cls.model_validate_json(Path(path).read_text())


class CSLRDataConfig(DataConfig):
    # Passed straight through to sldl.targets.ContinuousRecognitionTarget, which reads
    # sample["annotations"][annotation_id][column] as each window's gloss sequence.
    annotation_id: str = "both_hands"
    column: str = "lemma"
    # Path to a JSON file mapping gloss label -> class id (0 .. model.vocab_size - 1).
    label_to_id: str


class CSLRTrainConfig(StrictModel):
    data: CSLRDataConfig
    model: CSLRModelConfig = CSLRModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    trainer: TrainerConfig
    seed: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "CSLRTrainConfig":
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

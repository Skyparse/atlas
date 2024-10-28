# src/utils/config.py
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    # Model architecture parameters
    in_channels: int
    num_classes: int
    base_channel: int
    depth: int
    use_transformer: bool
    transformer_heads: int
    transformer_dim_head: int
    transformer_dropout: float
    use_fpn: bool
    fpn_channels: int
    deep_supervision: bool


@dataclass
class TrainingConfig:
    seed: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    mixed_precision: bool
    parallel: bool


@dataclass
class NormalizationConfig:
    mean: List[float]
    std: List[float]


@dataclass
class DataConfig:
    xA_path: str
    xB_path: str
    mask_path: Optional[str]
    val_split: float
    normalization: NormalizationConfig


@dataclass
class OutputConfig:
    output_dir: str
    experiment_name: str


@dataclass
class PredictionConfig:
    weights_path: str
    output_dir: str
    batch_size: int
    parallel: bool
    save_probabilities: bool


@dataclass
class TrainExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(
                **{
                    **config_dict["data"],
                    "normalization": NormalizationConfig(
                        **config_dict["data"]["normalization"]
                    ),
                }
            ),
            output=OutputConfig(**config_dict["output"]),
        )

    def validate(self):
        """Validate configuration parameters"""
        # Model validations
        assert self.model.in_channels > 0, "in_channels must be positive"
        assert self.model.num_classes > 0, "num_classes must be positive"
        assert self.model.base_channel > 0, "base_channel must be positive"
        assert self.model.depth > 0, "depth must be positive"

        # Training validations
        assert self.training.learning_rate > 0, "learning_rate must be positive"
        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.num_epochs > 0, "num_epochs must be positive"
        assert (
            self.training.gradient_accumulation_steps > 0
        ), "gradient_accumulation_steps must be positive"

        # Data validations
        assert Path(
            self.data.xA_path
        ).exists(), f"xA_path does not exist: {self.data.xA_path}"
        assert Path(
            self.data.xB_path
        ).exists(), f"xB_path does not exist: {self.data.xB_path}"
        if self.data.mask_path:
            assert Path(
                self.data.mask_path
            ).exists(), f"mask_path does not exist: {self.data.mask_path}"
        assert 0 < self.data.val_split < 1, "val_split must be between 0 and 1"


@dataclass
class PredictExperimentConfig:
    model: ModelConfig
    prediction: PredictionConfig
    data: DataConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PredictExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            prediction=PredictionConfig(**config_dict["prediction"]),
            data=DataConfig(
                **{
                    **config_dict["data"],
                    "normalization": NormalizationConfig(
                        **config_dict["data"]["normalization"]
                    ),
                    "mask_path": None,
                    "val_split": 0.0,  # Not used in prediction
                }
            ),
        )

    def validate(self):
        """Validate configuration parameters"""
        # Model validations
        assert self.model.in_channels > 0, "in_channels must be positive"
        assert self.model.num_classes > 0, "num_classes must be positive"
        assert self.model.base_channel > 0, "base_channel must be positive"
        assert self.model.depth > 0, "depth must be positive"

        # Prediction validations
        assert self.prediction.batch_size > 0, "batch_size must be positive"
        assert Path(
            self.prediction.weights_path
        ).exists(), f"weights_path does not exist: {self.prediction.weights_path}"

        # Data validations
        assert Path(
            self.data.xA_path
        ).exists(), f"xA_path does not exist: {self.data.xA_path}"
        assert Path(
            self.data.xB_path
        ).exists(), f"xB_path does not exist: {self.data.xB_path}"

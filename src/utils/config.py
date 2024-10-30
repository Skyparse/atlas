# src/utils/config.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    base_channel: int
    depth: int


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
    eval_frequency: int = 1


@dataclass
class LossConfig:
    focal_weight: float
    dice_weight: float
    focal_alpha: float
    focal_gamma: float


@dataclass
class NormalizationConfig:
    mean: List[float]
    std: List[float]


@dataclass
class BaseDataConfig:
    """Base data configuration shared between training and prediction"""

    xA_path: str
    xB_path: str
    normalization: NormalizationConfig


@dataclass
class TrainingDataConfig(BaseDataConfig):
    """Data configuration for training"""

    mask_path: str
    val_split: float


@dataclass
class PredictionDataConfig(BaseDataConfig):
    """Data configuration for prediction"""

    mask_path: Optional[str] = None


@dataclass
class OutputConfig:
    output_dir: str
    experiment_name: str


@dataclass
class PredictionConfig:
    weights_path: str
    output_dir: str
    batch_size: int
    parallel: bool = False
    save_individual: bool = False
    save_probabilities: bool = True


@dataclass
class TrainExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    data: TrainingDataConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            loss=LossConfig(**config_dict["loss"]),
            data=TrainingDataConfig(
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
        assert self.training.eval_frequency >= 0, "eval_frequency must be non-negative"

        # Data validations
        assert Path(
            self.data.xA_path
        ).exists(), f"xA_path does not exist: {self.data.xA_path}"
        assert Path(
            self.data.xB_path
        ).exists(), f"xB_path does not exist: {self.data.xB_path}"
        assert Path(
            self.data.mask_path
        ).exists(), f"mask_path does not exist: {self.data.mask_path}"
        assert 0 < self.data.val_split < 1, "val_split must be between 0 and 1"


@dataclass
class PredictExperimentConfig:
    model: ModelConfig
    prediction: PredictionConfig
    data: PredictionDataConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PredictExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            prediction=PredictionConfig(**config_dict["prediction"]),
            data=PredictionDataConfig(
                **{
                    **config_dict["data"],
                    "normalization": NormalizationConfig(
                        **config_dict["data"]["normalization"]
                    ),
                }
            ),
        )

    def validate(self):
        # Model validations
        assert self.model.in_channels > 0, "in_channels must be positive"
        assert self.model.num_classes > 0, "num_classes must be positive"
        assert self.model.base_channel > 0, "base_channel must be positive"
        assert self.model.depth > 0, "depth must be positive"

        # Prediction validations
        weights_path = Path(self.prediction.weights_path)
        assert weights_path.exists(), f"Weights file not found: {weights_path}"
        assert self.prediction.batch_size > 0, "batch_size must be positive"

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


@dataclass
class VisualizationStyle:
    """Configuration for visualization styling"""

    cmap: str = "hot"
    overlay_alpha: float = 0.5
    class_colors: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
        ]
    )
    figure_size: Tuple[int, int] = (15, 15)
    dpi: int = 300

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisualizationStyle":
        # Handle the old overlay_color key if present
        if "overlay_color" in config_dict:
            config_dict["class_colors"] = [config_dict.pop("overlay_color")]
        return cls(**config_dict)


@dataclass
class VisualizationOutput:
    """Configuration for visualization output"""

    save_dir: str
    num_samples: int = 10
    save_individual: bool = True
    save_summary: bool = True
    save_format: str = "png"
    create_animations: bool = False


@dataclass
class VisualizationOutput:
    """Configuration for visualization output"""

    save_dir: str
    num_samples: int = 10
    save_individual: bool = True
    save_summary: bool = True
    save_format: str = "png"
    create_animations: bool = False


@dataclass
class InputPaths:
    """Paths to input data"""

    predictions_path: str
    imageA_path: str
    imageB_path: str
    mask_path: Optional[str] = None


@dataclass
class VisualizationConfig:
    """Main configuration for visualization"""

    input: InputPaths
    output: VisualizationOutput
    style: VisualizationStyle = field(default_factory=VisualizationStyle)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisualizationConfig":
        return cls(
            input=InputPaths(**config_dict["input"]),
            output=VisualizationOutput(**config_dict["output"]),
            style=VisualizationStyle.from_dict(config_dict.get("style", {})),
        )

    def validate(self):
        """Validate visualization configuration"""
        # Validate input paths
        for path_name, path in {
            "predictions": self.input.predictions_path,
            "imageA": self.input.imageA_path,
            "imageB": self.input.imageB_path,
        }.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{path_name} file not found: {path}")

        if self.input.mask_path and not Path(self.input.mask_path).exists():
            raise FileNotFoundError(f"Mask file not found: {self.input.mask_path}")

        # Validate style parameters
        assert (
            0 <= self.style.overlay_alpha <= 1
        ), "overlay_alpha must be between 0 and 1"
        for colors in self.style.class_colors:
            for color in colors:
                assert 0 <= color <= 1, "color values must be between 0 and 1"

        # Validate output parameters
        assert self.output.num_samples > 0, "num_samples must be positive"
        assert self.style.dpi > 0, "dpi must be positive"

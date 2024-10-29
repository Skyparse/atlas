# src/utils/config.py
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from typing import Tuple


@dataclass
class ModelConfig:
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
    eval_frequency: int = 1


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
    data: TrainingDataConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainExperimentConfig":
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
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

    cmap: str = "hot"  # Colormap for prediction masks
    overlay_alpha: float = 0.5  # Transparency for overlays
    overlay_color: Tuple[float, float, float] = (1, 0, 0)  # RGB color for overlays
    figure_size: Tuple[int, int] = (15, 15)  # Size of individual figures
    dpi: int = 300  # DPI for saved figures


@dataclass
class VisualizationOutput:
    """Configuration for visualization output"""

    save_dir: str  # Directory to save visualizations
    num_samples: int = 10  # Number of samples to visualize
    save_individual: bool = True  # Save individual sample visualizations
    save_summary: bool = True  # Save summary figure
    save_format: str = "png"  # Format for saved figures
    create_animations: bool = False  # Create GIF animations of the change


@dataclass
class InputPaths:
    """Paths to input data"""

    predictions_path: str  # Path to prediction .npy file
    imageA_path: str  # Path to first timepoint images
    imageB_path: str  # Path to second timepoint images
    mask_path: Optional[str] = None  # Optional path to ground truth masks


@dataclass
class VisualizationConfig:
    """Main configuration for visualization"""

    input: InputPaths
    output: VisualizationOutput
    style: VisualizationStyle = VisualizationStyle()

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisualizationConfig":
        return cls(
            input=InputPaths(**config_dict["input"]),
            output=VisualizationOutput(**config_dict["output"]),
            style=VisualizationStyle(**config_dict.get("style", {})),
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
        for color in self.style.overlay_color:
            assert 0 <= color <= 1, "overlay_color values must be between 0 and 1"

        # Validate output parameters
        assert self.output.num_samples > 0, "num_samples must be positive"
        assert self.style.dpi > 0, "dpi must be positive"

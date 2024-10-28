# config.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class ModelConfig:
    # Model Architecture
    in_channels: int
    num_classes: int
    base_channel: int
    depth: int
    use_transformer: bool = True
    transformer_heads: int = 8
    transformer_dim_head: int = 64
    transformer_dropout: float = 0.1
    use_fpn: bool = True
    fpn_channels: int = 256
    deep_supervision: bool = True

    # Training Parameters
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_steps: int
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True

    # Optimization
    quantization: bool = False
    pruning: bool = False
    pruning_ratio: float = 0.3
    distillation: bool = False
    teacher_model_path: Optional[str] = None

    # Data Loading
    cache_size: int = 1000
    num_workers: int = 4
    prefetch_factor: int = 2

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def validate(self):
        """Validate configuration parameters"""
        assert self.in_channels > 0, "in_channels must be positive"
        assert self.num_classes > 0, "num_classes must be positive"
        assert self.base_channel > 0, "base_channel must be positive"
        assert self.depth > 0, "depth must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        if self.distillation:
            assert (
                self.teacher_model_path is not None
            ), "teacher_model_path required for distillation"

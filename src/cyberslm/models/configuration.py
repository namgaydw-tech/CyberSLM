from __future__ import annotations

from cyberslm.models.transformer import CyberSLM, ModelConfig, count_parameters
from cyberslm.utils.config import dataclass_from_yaml


def load_model_config(path: str) -> ModelConfig:
    return dataclass_from_yaml(ModelConfig, path)


def configured_parameter_count(config: ModelConfig) -> int:
    """Calculate the actual count; budget config filenames are only experiment labels."""
    return count_parameters(CyberSLM(config))

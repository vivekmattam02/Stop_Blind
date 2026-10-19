"""Model and training configuration."""

from spatialvlm.config.model import (
    BackboneConfig,
    EncoderConfig,
    FusionConfig,
    GeometryConfig,
    SpatialVLMConfig,
)
from spatialvlm.config.workshop import (
    WORKSHOP_CAUSAL_CHAIN_VERSION,
    WorkshopArmConfig,
    arm_factor_diff,
    build_workshop_causal_chain,
    validate_workshop_causal_chain,
)

__all__ = [
    "BackboneConfig",
    "EncoderConfig",
    "FusionConfig",
    "GeometryConfig",
    "SpatialVLMConfig",
    "WORKSHOP_CAUSAL_CHAIN_VERSION",
    "WorkshopArmConfig",
    "arm_factor_diff",
    "build_workshop_causal_chain",
    "validate_workshop_causal_chain",
]

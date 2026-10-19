"""Machine-readable C0-C4 causal contract for the workshop study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

WORKSHOP_CAUSAL_CHAIN_VERSION = 1


@dataclass(frozen=True)
class WorkshopArmConfig:
    """Scientific factors for one controlled workshop arm.

    This is an experiment registry, not yet an executed-model claim.  A runtime graph audit must
    prove that an implementation matches the selected record before its result is admissible.
    """

    arm_id: str
    history_contract_id: str
    vision_path: str
    visual_encoder: str
    scalar_metric_depth: bool
    explicit_metric_xyz: bool
    geometry_encoder: str
    positional_basis: str
    injection: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["causal_chain_version"] = WORKSHOP_CAUSAL_CHAIN_VERSION
        record["status"] = "proposed_until_runtime_graph_audit_passes"
        return record


def build_workshop_causal_chain(history_contract_id: str) -> tuple[WorkshopArmConfig, ...]:
    """Return the pre-registered C0-C4 intervention chain with one shared history contract."""
    if not isinstance(history_contract_id, str) or not history_contract_id.strip():
        raise ValueError("history_contract_id must be a non-empty frozen identifier.")
    history = history_contract_id.strip()
    c0 = WorkshopArmConfig(
        arm_id="C0",
        history_contract_id=history,
        vision_path="native_qwen3_vl",
        visual_encoder="native_qwen3_vl_vision",
        scalar_metric_depth=False,
        explicit_metric_xyz=False,
        geometry_encoder="none",
        positional_basis="native_qwen_mrope",
        injection="native_qwen_visual_path",
    )
    c1 = WorkshopArmConfig(
        arm_id="C1",
        history_contract_id=history,
        vision_path="external_spatial_tokens",
        visual_encoder="siglip2_plus_dinov2",
        scalar_metric_depth=True,
        explicit_metric_xyz=False,
        geometry_encoder="none",
        positional_basis="none_for_external_spatial_coordinates",
        injection="input_embedding_replacement",
    )
    c2 = replace(
        c1,
        arm_id="C2",
        explicit_metric_xyz=True,
        geometry_encoder="non_equivariant_point_transformer",
        positional_basis="cartesian_3d_rope",
    )
    c3 = replace(c2, arm_id="C3", geometry_encoder="gatr_projective_geometric_algebra")
    c4 = replace(c3, arm_id="C4", positional_basis="directional_metric_3d_rope")
    chain = (c0, c1, c2, c3, c4)
    validate_workshop_causal_chain(chain)
    return chain


def arm_factor_diff(left: WorkshopArmConfig, right: WorkshopArmConfig) -> tuple[str, ...]:
    """Return changed scientific factor names, excluding arm identity."""
    ignored = {"arm_id"}
    left_record = asdict(left)
    right_record = asdict(right)
    return tuple(
        key for key in left_record if key not in ignored and left_record[key] != right_record[key]
    )


def validate_workshop_causal_chain(chain: tuple[WorkshopArmConfig, ...]) -> None:
    """Fail closed if the proposed rows no longer support their intended contrasts."""
    if tuple(arm.arm_id for arm in chain) != ("C0", "C1", "C2", "C3", "C4"):
        raise ValueError("Workshop chain must contain exactly C0, C1, C2, C3, and C4 in order.")
    if len({arm.history_contract_id for arm in chain}) != 1:
        raise ValueError("Every C0-C4 arm must share one frozen history contract.")
    if arm_factor_diff(chain[2], chain[3]) != ("geometry_encoder",):
        raise ValueError("C3-C2 must isolate the geometry encoder.")
    if arm_factor_diff(chain[3], chain[4]) != ("positional_basis",):
        raise ValueError("C4-C3 must isolate the positional basis.")
    if not all(arm.scalar_metric_depth for arm in chain[1:]):
        raise ValueError("C1-C4 must receive the same scalar metric depth observation.")
    if not all(arm.explicit_metric_xyz for arm in chain[2:]):
        raise ValueError("C2-C4 must receive explicit calibrated metric XYZ.")

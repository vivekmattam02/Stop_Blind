"""Tests for the machine-readable C0-C4 causal chain."""

from spatialvlm.config.workshop import arm_factor_diff, build_workshop_causal_chain


def test_workshop_chain_preserves_shared_history_and_intended_adjacent_contrasts() -> None:
    chain = build_workshop_causal_chain("r2r_ce_h4_history_v1")

    assert [arm.arm_id for arm in chain] == ["C0", "C1", "C2", "C3", "C4"]
    assert {arm.history_contract_id for arm in chain} == {"r2r_ce_h4_history_v1"}
    assert arm_factor_diff(chain[2], chain[3]) == ("geometry_encoder",)
    assert arm_factor_diff(chain[3], chain[4]) == ("positional_basis",)


def test_workshop_chain_keeps_depth_and_xyz_matched_where_claimed() -> None:
    chain = build_workshop_causal_chain("r2r_ce_h1_history_v1")

    assert not chain[0].scalar_metric_depth
    assert all(arm.scalar_metric_depth for arm in chain[1:])
    assert not chain[1].explicit_metric_xyz
    assert all(arm.explicit_metric_xyz for arm in chain[2:])
    assert chain[0].vision_path == "native_qwen3_vl"
    assert all(arm.vision_path == "external_spatial_tokens" for arm in chain[1:])

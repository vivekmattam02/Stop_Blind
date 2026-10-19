"""Frozen allow-list for driving the saved native-C0 checkpoints in closed loop.

The locked closed-loop runner (``scripts/run_spark_native_c0_closed_loop.py``) can drive only the
six selected checkpoints in ``FROZEN_CHECKPOINTS``. The checkpoint sweep of
``PREREG_31_CHECKPOINTS.md`` (fifth draft and later) drives the other saved checkpoints with new,
additive code. This module is that code's scope: it names exactly 37 checkpoints (31 candidates and
the six selected comparators), the three suites (main, smoke and confirmation) and the four steps
(smoke, offline, main and confirmation). Each process drives or scores one checkpoint of one step;
nothing outside these lists can be driven.

Every checkpoint value below was copied from the section 3 table of the preregistration and checked
against each run's ``result.json`` (``checkpoints[]``, ``gate``, ``manifest_sha256``,
``selection``). The runner re-checks all of them against the artifacts before anything is loaded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from spatialvlm.eval.native_closed_loop import FROZEN_CHECKPOINTS, ClosedLoopCheckpoint

ALLOWLIST_SCHEMA_VERSION = 2
EPISODES_PER_SUITE = 61
LOCKED_RUN_ID = "native_c0_closed_loop_v1"
# Sections 4.1 and 4.3: the simulator runs on GPU device 0, the job's one GPU.
GPU_DEVICE_ID = 0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Run ids: native_c0_sweep_torch_<step>_v<k>. A run that moves to another GPU type (section 4.2)
# reruns under the next k; the smoke may run several times under one k (section 5, step 2).
RUN_ID_RE = re.compile(r"^native_c0_sweep_torch_(smoke|offline|main|confirmation)_v([1-9][0-9]*)$")
# Section 6: the smoke drives wt25-u192, and its first worker stops after three completed
# episodes (episode 10) and resumes in a new invocation that starts with episode 11.
SMOKE_CHECKPOINT_ID = "inverse_frequency_s20260825_u192"
SMOKE_STOP_AFTER_COMPLETED_EPISODES = 3


class SweepRefusalError(RuntimeError):
    """Raised when a request falls outside the frozen sweep scope. The message names the reason."""


@dataclass(frozen=True)
class SweepRun:
    """One of the six C0 training runs, as the preregistration's run table records it."""

    tag: str
    treatment: str
    replication_seed: int
    source_run_id: str
    checkpoint_gate: str
    source_manifest_sha256: str
    selected_update: int
    legacy_checkpoint_id: str
    # The run's result.json and verification.json, pinned to the hashes that
    # native_c0_closed_loop_v1 recorded for them (its manifest's checkpoint_locks).
    source_result_sha256: str
    source_verification_sha256: str


@dataclass(frozen=True)
class SweepCheckpoint:
    """One checkpoint the sweep may drive.

    ``checkpoint_relative_path`` is relative to the run's archive directory, exactly as the run's
    ``result.json`` records it and as ``FROZEN_CHECKPOINTS`` stores it, so the locked
    ``_checkpoint_path(archive, record)`` resolves it unchanged.
    """

    checkpoint_id: str
    prereg_tag: str
    prereg_row: int
    treatment: str
    replication_seed: int
    source_run_id: str
    checkpoint_gate: str
    source_manifest_sha256: str
    update_step: int
    checkpoint_relative_path: str
    checkpoint_sha256: str
    trainable_state_sha256: str
    role: str
    legacy_checkpoint_id: str | None

    @property
    def archive_relative_path(self) -> str:
        return f"{self.source_run_id}/{self.checkpoint_relative_path}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["archive_relative_path"] = self.archive_relative_path
        return payload

    def as_frozen_checkpoint(self) -> ClosedLoopCheckpoint:
        """Return this record in the locked ``ClosedLoopCheckpoint`` form (selected rows only)."""
        if self.role != "selected" or self.legacy_checkpoint_id is None:
            raise SweepRefusalError(
                f"{self.checkpoint_id} is not one of the six selected checkpoints."
            )
        return ClosedLoopCheckpoint(
            checkpoint_id=self.legacy_checkpoint_id,
            treatment=self.treatment,
            replication_seed=self.replication_seed,
            source_run_id=self.source_run_id,
            checkpoint_gate=self.checkpoint_gate,
            selected_update=self.update_step,
            checkpoint_relative_path=self.checkpoint_relative_path,
            checkpoint_sha256=self.checkpoint_sha256,
        )


RUNS: tuple[SweepRun, ...] = (
    SweepRun(
        tag="uw25",
        treatment="unweighted",
        replication_seed=20260825,
        source_run_id="native_c0_development_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        source_manifest_sha256="726952465b43827384ecb6bc20a41d867fbb83a3027c4cae30ebe00ccb38ab7b",
        selected_update=160,
        legacy_checkpoint_id="unweighted_s20260825",
        source_result_sha256="88bbd74f41eb6bb9d673e12fb3aa094accc5b4a93e5e6c0fe81c8352a3339c61",
        source_verification_sha256="4f9116045645ecfbd84638f09d3e94b0db50b77d42d7a3cafdba6bbcfae230c8",
    ),
    SweepRun(
        tag="wt25",
        treatment="inverse_frequency",
        replication_seed=20260825,
        source_run_id="native_c0_action_balance_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        source_manifest_sha256="056284189286c66de34fbe0dd5b4b9f59ecc1d805688a49440692cfa5548867a",
        selected_update=192,
        legacy_checkpoint_id="inverse_frequency_s20260825",
        source_result_sha256="ced3e7c213508bf0c80f1aeed60e732aaf71f2cd133eb056e716305a2a089866",
        source_verification_sha256="2b572451daaf07bf64c6053229edaf059e0ffd5bba2a3bb6d762fdc3c71f18dd",
    ),
    SweepRun(
        tag="uw26",
        treatment="unweighted",
        replication_seed=20260826,
        source_run_id="native_c0_seed_replication_unweighted_s20260826_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        source_manifest_sha256="a6a2971b11aea148fa2158b02166f88bce1c5845a6496013e89bb0b61d8fdfb0",
        selected_update=160,
        legacy_checkpoint_id="unweighted_s20260826",
        source_result_sha256="946d05b33a7dc860c287be6b4675507767693a4a13191b63cd6aff872b497c69",
        source_verification_sha256="e81dd23bb648976be0c2b02776fe34611ed9db0cd117051007f2a857f277b859",
    ),
    SweepRun(
        tag="wt26",
        treatment="inverse_frequency",
        replication_seed=20260826,
        source_run_id="native_c0_seed_replication_inverse_frequency_s20260826_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        source_manifest_sha256="d78f32d8daa0548cdbe0feaacf2577b6dd488110119fcd648a248a74a1ccebee",
        selected_update=128,
        legacy_checkpoint_id="inverse_frequency_s20260826",
        source_result_sha256="52ae0f71a5269ed92794b0412f87af9391ee25bff749b5c1dd553a18cc6f8aa0",
        source_verification_sha256="663e721391f27347fc773013c82b281201fe933b9391b86ae601bec883135971",
    ),
    SweepRun(
        tag="uw27",
        treatment="unweighted",
        replication_seed=20260827,
        source_run_id="native_c0_seed_replication_unweighted_s20260827_v1",
        checkpoint_gate="native_qwen_c0_development_v1",
        source_manifest_sha256="be24ec75a8913e7b527a84ecfb43d5e6aff143e52aa87f1b2dd7a4fc66d9117e",
        selected_update=160,
        legacy_checkpoint_id="unweighted_s20260827",
        source_result_sha256="65082403f55eda913fc9967e3dc37c1f578e999353a9afd12490c1b25370e139",
        source_verification_sha256="c23ceb4f0008b694b93249640554bc370e8ef1e54c0aa03c75279db2c2d3b730",
    ),
    SweepRun(
        tag="wt27",
        treatment="inverse_frequency",
        replication_seed=20260827,
        source_run_id="native_c0_seed_replication_inverse_frequency_s20260827_v1",
        checkpoint_gate="native_qwen_c0_action_balance_v1",
        source_manifest_sha256="61312cc64871b81b0fc51f20681f85636b04b223539a60253bf5d1f6a47385fd",
        selected_update=192,
        legacy_checkpoint_id="inverse_frequency_s20260827",
        source_result_sha256="46fe665925d94fd4070ee94d97b0b1e7657a214d78d4b9c2e0f8d6f2ebb2978c",
        source_verification_sha256="4983a8b59a8caaa3eba0b885bc1d0d08762d18ca9ecebedd25cc94023969d4bf",
    ),
)
_RUN_BY_TAG = {run.tag: run for run in RUNS}
_UPDATES = (0, 32, 64, 96, 128, 160, 192)


def _checkpoint_id(treatment: str, seed: int, update: int) -> str:
    return f"{treatment}_s{seed}_u{update:03d}"


def _record(
    tag: str, update: int, checkpoint_sha256: str, trainable_state_sha256: str, role: str
) -> SweepCheckpoint:
    run = _RUN_BY_TAG[tag]
    run_index = RUNS.index(run)
    selected = role == "selected"
    if selected != (update == run.selected_update):
        raise ValueError(
            f"Allow-list role disagrees with the run's selected update: {tag}-{update}"
        )
    return SweepCheckpoint(
        checkpoint_id=_checkpoint_id(run.treatment, run.replication_seed, update),
        prereg_tag=f"{tag}-u{update}",
        prereg_row=run_index * len(_UPDATES) + _UPDATES.index(update) + 1,
        treatment=run.treatment,
        replication_seed=run.replication_seed,
        source_run_id=run.source_run_id,
        checkpoint_gate=run.checkpoint_gate,
        source_manifest_sha256=run.source_manifest_sha256,
        update_step=update,
        checkpoint_relative_path=f"step_{update:04d}.pt",
        checkpoint_sha256=checkpoint_sha256,
        trainable_state_sha256=trainable_state_sha256,
        role=role,
        legacy_checkpoint_id=run.legacy_checkpoint_id if selected else None,
    )


# The 37 rows of preregistration section 3 marked "drive" (31) or "selected" (6), in table order.
# fmt: off
ALLOWED_CHECKPOINTS: tuple[SweepCheckpoint, ...] = (
    _record("uw25", 0, "2dcae1f8297c09d34b853892191078b0babd801809d2cce757ea0fd77e9e2c66", "e7437fb6ee1f20411906d09b1c4ff072b905c1578798ad0ac2ebb314b753aa25", "candidate"),  # noqa: E501
    _record("uw25", 32, "d75b372f5d68db92a2b870a283b8545bd3246b7316b7b88cf9d4c692d20d789c", "f2fc80acd73992e03e9db6db107a6875fe594ce9a274267ab457e95d77429f37", "candidate"),  # noqa: E501
    _record("uw25", 64, "95853ea867f6802038153d6a78b229d817db2b95e67ba28088eb04ee8a22725c", "059743c7bdd77c586f077c6523647c0f26bb08687f3f84b11f049a1e66100c93", "candidate"),  # noqa: E501
    _record("uw25", 96, "fd52bd1c84f9e180a29a1709424a1c25be348cd96af0cbc62dbcc94e9a4c0cd0", "7d8b7b1cbc29d36b3d1ee61bd5c63a298e93efcc37b02ad4f490439dd77406fd", "candidate"),  # noqa: E501
    _record("uw25", 128, "f9ef8df099ee907e1f1b5a1005d5314ba2c61a444fae9ee3d5c67e9edcb8c9ad", "5a0f18addd2e995f45baf3fa5f426140410df708cebef52f5eed9a9be62d8bfe", "candidate"),  # noqa: E501
    _record("uw25", 160, "a4a58cd34409e24c805c847249cd921da17d11c2bef96e37a20bf7c50a86987d", "1bbb8490f51ca79c5109058a943cfa5c011de9ce69cfaf088dcdff672d5d984f", "selected"),  # noqa: E501
    _record("uw25", 192, "566b58d110e1196c1a94c1739df6bba518115d21d675726d05c2f8e4d570f68d", "518bcc8f4f51fc4f96399d5b670f7634df3d05f1f79af7fe9773d13d6297c14c", "candidate"),  # noqa: E501
    _record("wt25", 32, "e884bca4685fe6ebe4b3463cfeefa6c9e5976ebe84dad150a67a48c957ae5f9f", "b4105798c599e205541ab5efcc37eea3b497ca37b051f4c89b6b9c1bd9e3de85", "candidate"),  # noqa: E501
    _record("wt25", 64, "8aad706e03b626282f14418fe464412d0df2c898e7b42926ed54d84601b67236", "f492d61eeff338ac624c2a287ad1a5eba9b1a4bf3370c0a75af21ac1968a94b0", "candidate"),  # noqa: E501
    _record("wt25", 96, "27d036a50d1b5be4ef5a9cfcf09bfc5d36e44a20cce9da96aa25df3d7a5acf7d", "486a1c6910b39a44e03d79fe9c20257083074d0eec930650b6d553543ddbdca7", "candidate"),  # noqa: E501
    _record("wt25", 128, "717750ec416fd4f67281cda0eefe38141a0c6ad31a13905eeb112971a7dae63c", "a0958939a7db9297e06e03956279ef69b82df44af9fcbe26fe97479af75fdf80", "candidate"),  # noqa: E501
    _record("wt25", 160, "7ef6df2cc477fded67cbd54337feec4b10f591424cd67a06ee9bd7de96e21fb0", "e2430c68ea2bdc881a889e4f3de706d3a94e6c02061bbbe4999a05b9453df976", "candidate"),  # noqa: E501
    _record("wt25", 192, "f21496af3bc35bcc9f680b142bb19b4eff7f92fb5e8b405837e50ff7181872dc", "67381fcef28903f537e985a064a6ab9088dde0764f6f5fcd51bc10efb71cbbbc", "selected"),  # noqa: E501
    _record("uw26", 32, "ecc23c53ec8756053f36938c881d60e47a8e329ef89d480566106ed49e075886", "73ebeb0fe42bc55ca31313c0d629f6886ec019836abe237c4ab74b724420fa38", "candidate"),  # noqa: E501
    _record("uw26", 64, "c259b21a148bcfa21caa0218a9964e344571b79b9574914dc74ca19366cda280", "d7af475b1f353fb6f5e49b580788ced9c9034f2f0c25db56d1da1770cf3e19c1", "candidate"),  # noqa: E501
    _record("uw26", 96, "a01ec71931282de2ba8c3422ff2b7e994d3efc130b49ee52c1cd6eee6a5af4f4", "f8bab93e16d7d817b3e4058bf02e8166562bde4675a3ba048b94f382744d4c54", "candidate"),  # noqa: E501
    _record("uw26", 128, "53cff620c3a2aa677d9f5afde78c1a2bb4fade241f4c87375339d281d3b08228", "1f4a7deb612b96e01d504d78f6a42545bc9d4d3e2ed6ce7e523e088d97a13715", "candidate"),  # noqa: E501
    _record("uw26", 160, "ac5b106093c3b7e0a30851f907bba094b0db35f382940394bbc6d95fd4b89413", "f9e43568acfe84bdef1ae311d3bed6beb7f47160d1ae9f5d43b0c4840a265c53", "selected"),  # noqa: E501
    _record("uw26", 192, "247c203c04f2ddc266dbb7fbbf5122f6d3f271938d9f0b3437d631307457645b", "950eeee6a18804e87fc3ae4bb61502a510df7138a9ea803206cd11d94ab3e0e5", "candidate"),  # noqa: E501
    _record("wt26", 32, "6f3c44e8aa57358e766007c1912a6b8cf305a2c6ad02b14de52fb62642fe4b20", "0f5832372d7dce4d2bdedec32c8b4f642430bc49301715b8ec0cb45d80c48b1f", "candidate"),  # noqa: E501
    _record("wt26", 64, "69e2be613e1708295c85f5c7afb3e4452cdddcd5362d5b6bd9be6f81640d1f30", "fd3ba132bde955f7bc2b7abd57d2305afc2a0565c57bb18ab0c36f0a609d9b0b", "candidate"),  # noqa: E501
    _record("wt26", 96, "84ecf502c1a24f8f52678ed054a5b65ad8143695712b725dc6be132b1f43d4e3", "8f8d2966c77f2ca19f2dbd7a6d382fff9f99a2700ea542dddf13df5bb0fce991", "candidate"),  # noqa: E501
    _record("wt26", 128, "55380852648df9581c06b887f5399240f981245d4da7b868019fb6ea8a544260", "71f9a4bdc688b3bf2914d9c7f584d32e57f593d0f8e7f509b6a895c06a3c80c8", "selected"),  # noqa: E501
    _record("wt26", 160, "18626bea33bb083a7b11a140cfdd5223e108701589f6e956484c737c6a9cd8a0", "dac2553daff6880ceacb362203ca2049363c098e4ac41078c297df4eadc91543", "candidate"),  # noqa: E501
    _record("wt26", 192, "f04d62da03568bf46e586189a310ccdcdf14aabcb595e265d1eca370649e5f98", "3264315d412e330185ce054885f28a26d8ae74abbc3ef1ef0f0d2964785e37bd", "candidate"),  # noqa: E501
    _record("uw27", 32, "315ace9c9cf3ddf43e21d8d28385dc0ae1efd941b060661104f025f227928a52", "008e8699736f549f900e58a2662660cce7dec74a32092c3267c1fd1275c5fcbd", "candidate"),  # noqa: E501
    _record("uw27", 64, "84257d577bac7aa5265a04b8a9cf9c6e4af662befa774979c2cad68b0be07b1a", "3654a1d6cc5d460825c625a9ded1dab46503be539a6883e7d22e4706c7fc5ae5", "candidate"),  # noqa: E501
    _record("uw27", 96, "fb935b2d21b90cf19afc4e7ccc4962847d2763f185f9218964ec98c4f0cb0e5e", "2ffaf19251683bd47714b42e14c678a9e99a9f77595a1b72cb6fa9794c0af224", "candidate"),  # noqa: E501
    _record("uw27", 128, "5ad08189aad45d6a2bd4d712a2a4658a1d0221f791937ed1e63dc501517fcad4", "03d49dd55527f6450a42e0c0dd9e55426f67c226013790997be01a0da83f9292", "candidate"),  # noqa: E501
    _record("uw27", 160, "7264fdd47d9554ce9d21da8f29d7ccb73dbe3903560b587c26106297cf86da21", "9e56e84b11eeecacb6c8b8186a0e74e4d3d99f39933be61baa6b826a00a83fba", "selected"),  # noqa: E501
    _record("uw27", 192, "e344473921433c1e7a3f34abbb3894733ac0f162394fb237dc94b1af95756aa3", "8f354bd0932174dfbeb573c20c98e098a8b338604ff155c1651a77c518d23d52", "candidate"),  # noqa: E501
    _record("wt27", 32, "119ddddda95f6375878373f6b00cda6db88bd97009d1f4460ca6e19cd449e0f7", "ed499aafb2ed2da3a61a2a979569fa8fb1511bd7bdc60587e84627b1f3a0565e", "candidate"),  # noqa: E501
    _record("wt27", 64, "c7f9788a1d10d0471c75327ff0d2cff9a7c3992c3685adc0de8472a1f1c38c3b", "27821afcfb804eeebe66a89dce953feda59fe521267cca9521a47ee4cc4bef53", "candidate"),  # noqa: E501
    _record("wt27", 96, "e4c1d0fc52aeb69dd98f72952436c1ba3d349fecc087d983eac63154fa1ece17", "fbbcd331ee8330af693c3bdcedab7c557b3170196e6b6bc187a0db715d2ea70a", "candidate"),  # noqa: E501
    _record("wt27", 128, "f9e1418ab8bfdff4ae83402f4c96754227f04431feee28242b26574de623db63", "bb0fd4501708582cd3f0f4c09295e4073b1779c888daaba2cb0ec723ff518928", "candidate"),  # noqa: E501
    _record("wt27", 160, "ca3751c1bcfd5908f0914e3a48d866d364a63f6db2d8219b1589d86e9905aa71", "3ce46b906e29b2e104f33fabf7fb034976e37bcf62d86458570ddedf66281450", "candidate"),  # noqa: E501
    _record("wt27", 192, "77f6eb8695323db53cf7394125f33a4a78be0b9608c500dfe575b4078eec81fc", "36c807329351d3c60882a4728e7edfec7830c7cb57f440d32f2b065854b3639d", "selected"),  # noqa: E501
)

# The five update-0 files that are refused. Each is the frozen base model (every lora_B tensor is
# exactly zero), and unweighted_s20260825_u000 is driven once to stand for all six.
REFUSED_UPDATE_ZERO_DUPLICATES: tuple[tuple[str, str, str], ...] = (
    ("native_c0_action_balance_v1", "step_0000.pt", "7ea1269caf1ba2d7b5e1542c31d4a6819efa76b4d1330ba40528b016ccd38c1f"),  # noqa: E501
    ("native_c0_seed_replication_unweighted_s20260826_v1", "step_0000.pt", "d361ac03db673243fc1c66b056c40b4d1717087f8cf0adca54d01ca0bb4d38c1"),  # noqa: E501
    ("native_c0_seed_replication_inverse_frequency_s20260826_v1", "step_0000.pt", "0e3546adfee4d2dca3dff789345eca1c07c696c97522851d1f5c362d3b078861"),  # noqa: E501
    ("native_c0_seed_replication_unweighted_s20260827_v1", "step_0000.pt", "1643dd842258667e006990ca6e4cae5294d3793bbe947dd7e197b92fa6203132"),  # noqa: E501
    ("native_c0_seed_replication_inverse_frequency_s20260827_v1", "step_0000.pt", "54004b26549e2ccea838fa7230dc2de2473fa3f48b111ff7060043115643f53d"),  # noqa: E501
)
# fmt: on

_BY_ID = {record.checkpoint_id: record for record in ALLOWED_CHECKPOINTS}
_REFUSED_IDS = {
    _checkpoint_id(run.treatment, run.replication_seed, 0): run.source_run_id
    for run in RUNS
    if run.tag != "uw25"
}
UPDATE_ZERO_DRIVEN_ID = "unweighted_s20260825_u000"
CANDIDATE_IDS: tuple[str, ...] = tuple(
    record.checkpoint_id for record in ALLOWED_CHECKPOINTS if record.role == "candidate"
)
SELECTED_IDS: tuple[str, ...] = tuple(
    record.checkpoint_id for record in ALLOWED_CHECKPOINTS if record.role == "selected"
)
# H3 (section 7.6): the 15 trained unweighted checkpoints that were not selected.
UNWEIGHTED_UNSELECTED_TRAINED_IDS: tuple[str, ...] = tuple(
    record.checkpoint_id
    for record in ALLOWED_CHECKPOINTS
    if record.treatment == "unweighted" and record.role == "candidate" and record.update_step > 0
)
RUN_TAGS: tuple[str, ...] = tuple(run.tag for run in RUNS)
UPDATES: tuple[int, ...] = _UPDATES


def run_by_tag(tag: str) -> SweepRun:
    """Return one of the six runs by its section 3 tag."""
    return _RUN_BY_TAG[tag]


def run_of(record: SweepCheckpoint) -> SweepRun:
    """Return the run a checkpoint belongs to."""
    for run in RUNS:
        if run.source_run_id == record.source_run_id:
            return run
    raise AssertionError(record.checkpoint_id)  # pragma: no cover


def checkpoint_for(tag: str, update: int) -> SweepCheckpoint:
    """Return the record that stands for update ``update`` of run ``tag``.

    Update 0 is the same function in every run (every ``lora_B`` is zero), so it is always the
    one driven update-0 file, uw25-u0 (section 3, "Eligibility" in section 8).
    """
    if update == 0:
        return _BY_ID[UPDATE_ZERO_DRIVEN_ID]
    run = _RUN_BY_TAG[tag]
    return _BY_ID[_checkpoint_id(run.treatment, run.replication_seed, update)]


@dataclass(frozen=True)
class SweepSuite:
    """One of the three suites. ``sha256`` is the sha256 of the suite file the runner reads:
    the control manifest for the main suite, the committed suite file for the other two."""

    suite_id: str
    description: str
    sha256: str
    path: str | None
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUITES: dict[str, SweepSuite] = {
    "main": SweepSuite(
        suite_id="main",
        description=(
            "the 61 entries of artifacts/spark/policy_ordering_v1/suite_manifest.json, read as "
            "control_manifest['episodes'][:61]"
        ),
        sha256="cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4",
        path=None,
        episode_count=EPISODES_PER_SUITE,
    ),
    "smoke": SweepSuite(
        suite_id="smoke",
        description=(
            "section 6: val_seen episodes 4, 5, 10, 11, 13, 14, 16 and 17, in neither the main "
            "nor the confirmation suite, built by scripts/build_native_c0_confirmation_suite.py"
        ),
        sha256="bd4d681a807ffa92e2963b5fd6b3ecf60a5857aea8c85e8a61d99c94e765961c",
        path="configs/stopblind/smoke_suite.json",
        episode_count=8,
    ),
    "confirmation": SweepSuite(
        suite_id="confirmation",
        description=(
            "section 7.5: 61 val_seen trajectories outside the main suite drawn with "
            "random.Random(20260930), built by scripts/build_native_c0_confirmation_suite.py"
        ),
        sha256="a34950cdb7fa0855066a2568cf2a5bc0a9631684a94fb5e0ca0ab79b749f88f5",
        path="configs/stopblind/confirmation_suite.json",
        episode_count=EPISODES_PER_SUITE,
    ),
}


@dataclass(frozen=True)
class SweepStep:
    """One step of the sweep: which programs run it, on which suite, over which checkpoints."""

    step: str
    programs: tuple[str, ...]
    suite_id: str | None
    checkpoint_ids: tuple[str, ...]
    narrowed_at_run_time: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["programs"] = list(self.programs)
        payload["checkpoint_ids"] = list(self.checkpoint_ids)
        return payload


ALL_IDS: tuple[str, ...] = tuple(record.checkpoint_id for record in ALLOWED_CHECKPOINTS)
RUNNER = "runner"
SCORER = "scorer"
STEPS: dict[str, SweepStep] = {
    "smoke": SweepStep(
        step="smoke",
        programs=(RUNNER, SCORER),
        suite_id="smoke",
        checkpoint_ids=(SMOKE_CHECKPOINT_ID,),
        narrowed_at_run_time=False,
        description=(
            "section 6: wt25-u192 on the eight smoke episodes, and the scorer's check on its "
            "training records; before the freeze commit"
        ),
    ),
    "offline": SweepStep(
        step="offline",
        programs=(SCORER,),
        suite_id=None,
        checkpoint_ids=ALL_IDS,
        narrowed_at_run_time=False,
        description=(
            "section 8: score all 37 policies on the selection, natural and balanced decisions, "
            "on the study GPU, before the main rollouts"
        ),
    ),
    "main": SweepStep(
        step="main",
        programs=(RUNNER,),
        suite_id="main",
        checkpoint_ids=ALL_IDS,
        narrowed_at_run_time=False,
        description="section 5: all 37 policies on the 61 main episodes",
    ),
    "confirmation": SweepStep(
        step="confirmation",
        programs=(RUNNER,),
        suite_id="confirmation",
        checkpoint_ids=CANDIDATE_IDS + SELECTED_IDS,
        narrowed_at_run_time=True,
        description=(
            "section 7.5: at most three carried candidates and their comparators, narrowed at "
            "run time by the carried list the analysis wrote"
        ),
    ),
}


def refuse(message: str) -> SweepRefusalError:
    """Build the refusal exception with a uniform prefix."""
    return SweepRefusalError(f"Refused: {message}")


def step_record(step: str) -> SweepStep:
    """Return the frozen record of one step, refusing any other name."""
    if step not in STEPS:
        raise refuse(f"step {step!r} is not one of {sorted(STEPS)}.")
    return STEPS[step]


def check_step_program(step: str, program: str) -> SweepStep:
    """Refuse a step that this program does not run."""
    record = step_record(step)
    if program not in record.programs:
        raise refuse(f"the {program} does not run step {step}.")
    return record


def resolve_checkpoint(
    checkpoint_id: str,
    *,
    source_run_id: str | None = None,
    update_step: int | None = None,
    checkpoint_relative_path: str | None = None,
) -> SweepCheckpoint:
    """Return the allow-listed record for ``checkpoint_id``.

    Refuses an id that is not one of the 37 (naming the update-0 duplicates explicitly), and a
    listed id whose run, update or file differs from the record.
    """
    if checkpoint_id in _REFUSED_IDS:
        raise refuse(
            f"{checkpoint_id} is an update-0 duplicate and is not driven; "
            f"{UPDATE_ZERO_DRIVEN_ID} stands for all six update-0 files."
        )
    record = _BY_ID.get(checkpoint_id)
    if record is None:
        raise refuse(f"checkpoint {checkpoint_id!r} is not one of the 37 allow-listed checkpoints.")
    mismatches = {}
    if source_run_id is not None and source_run_id != record.source_run_id:
        mismatches["source_run_id"] = (source_run_id, record.source_run_id)
    if update_step is not None and (
        isinstance(update_step, bool) or update_step != record.update_step
    ):
        mismatches["update_step"] = (update_step, record.update_step)
    if (
        checkpoint_relative_path is not None
        and checkpoint_relative_path != record.checkpoint_relative_path
    ):
        mismatches["checkpoint_relative_path"] = (
            checkpoint_relative_path,
            record.checkpoint_relative_path,
        )
    if mismatches:
        raise refuse(
            f"{checkpoint_id} was requested with values the allow-list does not hold: "
            f"{mismatches} (requested, listed)."
        )
    return record


def resolve_checkpoint_file(source_run_id: str, checkpoint_relative_path: str) -> SweepCheckpoint:
    """Return the record for one archive file, refusing the update-0 duplicates and others."""
    for run_id, relative, _sha in REFUSED_UPDATE_ZERO_DUPLICATES:
        if (run_id, relative) == (source_run_id, checkpoint_relative_path):
            raise refuse(
                f"{source_run_id}/{checkpoint_relative_path} is an update-0 duplicate and is not "
                f"driven; {UPDATE_ZERO_DRIVEN_ID} stands for all six update-0 files."
            )
    for record in ALLOWED_CHECKPOINTS:
        if (record.source_run_id, record.checkpoint_relative_path) == (
            source_run_id,
            checkpoint_relative_path,
        ):
            return record
    raise refuse(f"{source_run_id}/{checkpoint_relative_path} is not an allow-listed checkpoint.")


def check_run_id(step: str, run_id: str) -> int:
    """Refuse the locked run id and any run id that is not the step's; return its version k."""
    if run_id == LOCKED_RUN_ID:
        raise refuse(f"run id {LOCKED_RUN_ID} belongs to the locked closed loop.")
    step_record(step)
    match = RUN_ID_RE.fullmatch(run_id)
    if match is None or match.group(1) != step:
        raise refuse(
            f"step {step} must use a run id native_c0_sweep_torch_{step}_v<k>, not {run_id!r}."
        )
    return int(match.group(2))


def comparators_of(checkpoint_id: str) -> tuple[SweepCheckpoint, ...]:
    """Return the section 7.2 comparators of one candidate, in section 3 order.

    A trained candidate has one: its own run's selected checkpoint. Update 0 stands for the
    update-0 file of all six runs, so it has six: every selected checkpoint. Its H1 and
    confirmation tests are intersection-union tests: it must beat each of the six.
    """
    record = resolve_checkpoint(checkpoint_id)
    if record.role != "candidate":
        raise refuse(f"{checkpoint_id} is a selected checkpoint, not a candidate.")
    if record.checkpoint_id == UPDATE_ZERO_DRIVEN_ID:
        return tuple(other for other in ALLOWED_CHECKPOINTS if other.role == "selected")
    for other in ALLOWED_CHECKPOINTS:
        if other.role == "selected" and other.source_run_id == record.source_run_id:
            return (other,)
    raise AssertionError(f"No selected checkpoint for {record.source_run_id}.")  # pragma: no cover


def comparator_of(checkpoint_id: str) -> SweepCheckpoint:
    """Return the one section 7.2 comparator of a trained candidate: its run's selected checkpoint.

    Update 0 has six comparators (``comparators_of``), so it is refused here.
    """
    comparators = comparators_of(checkpoint_id)
    if len(comparators) != 1:
        raise refuse(
            f"{checkpoint_id} has {len(comparators)} comparators (section 7.2); use comparators_of."
        )
    return comparators[0]


def frozen_checkpoint_for(record: SweepCheckpoint) -> ClosedLoopCheckpoint:
    """Return the locked ``FROZEN_CHECKPOINTS`` entry this selected record must equal."""
    matches = [
        item for item in FROZEN_CHECKPOINTS if item.checkpoint_id == record.legacy_checkpoint_id
    ]
    if record.role != "selected" or len(matches) != 1:
        raise refuse(f"{record.checkpoint_id} has no FROZEN_CHECKPOINTS entry.")
    return matches[0]


def ordered(checkpoint_ids: Sequence[str]) -> tuple[SweepCheckpoint, ...]:
    """Return the records in allow-list order (section 3 order)."""
    wanted = set(checkpoint_ids)
    return tuple(record for record in ALLOWED_CHECKPOINTS if record.checkpoint_id in wanted)


def confirmation_scope_from_carried_list(carried: Mapping[str, Any]) -> tuple[SweepCheckpoint, ...]:
    """Return the confirmation scope a carried list justifies: its candidates and comparators.

    Expected form: ``{"carried": [{"checkpoint_id": ..., "source_run_id": ...,
    "update_step": ..., "comparator_checkpoint_ids": [...], "comparator_checkpoint_id": ...},
    ...]}``, at most three entries, in the analysis script's carry order. Both comparator keys are
    optional; ``comparator_checkpoint_ids`` must list the section 7.2 comparators exactly, and the
    single ``comparator_checkpoint_id`` is allowed only for a candidate with one comparator (update
    0 has six). Any other shape, an unlisted or selected candidate, a wrong update, or a comparator
    other than the section 7.2 ones is refused.
    """
    entries = carried.get("carried")
    if not isinstance(entries, list):
        raise refuse("the carried list has no 'carried' list.")
    if not entries:
        raise refuse("the carried list is empty, so the confirmation step does not run.")
    if len(entries) > 3:
        raise refuse(
            f"the carried list names {len(entries)} candidates; at most three are carried."
        )
    scope: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("checkpoint_id"), str):
            raise refuse(f"carried entry is malformed: {entry!r}")
        record = resolve_checkpoint(
            entry["checkpoint_id"],
            source_run_id=entry.get("source_run_id"),
            update_step=entry.get("update_step"),
        )
        if record.role != "candidate":
            raise refuse(f"carried entry {record.checkpoint_id} is not one of the 31 candidates.")
        comparators = [item.checkpoint_id for item in comparators_of(record.checkpoint_id)]
        named = entry.get("comparator_checkpoint_id")
        if named is not None and [named] != comparators:
            raise refuse(
                f"carried entry {record.checkpoint_id} names comparator {named!r}; its section "
                f"7.2 comparators are {comparators}."
            )
        named_list = entry.get("comparator_checkpoint_ids")
        if named_list is not None and list(named_list) != comparators:
            raise refuse(
                f"carried entry {record.checkpoint_id} names comparators {named_list!r}; its "
                f"section 7.2 comparators are {comparators}."
            )
        if record.checkpoint_id in scope:
            raise refuse(f"carried entry {record.checkpoint_id} appears twice.")
        scope.append(record.checkpoint_id)
        for comparator in comparators:
            if comparator not in scope:
                scope.append(comparator)
    return ordered(scope)


def step_scope(
    step: str, *, carried: Mapping[str, Any] | None = None
) -> tuple[SweepCheckpoint, ...]:
    """Return the ordered checkpoints of one step; the confirmation step needs the carried list."""
    frozen = step_record(step)
    if frozen.narrowed_at_run_time:
        if carried is None:
            raise refuse(f"step {step} requires the carried list.")
        return confirmation_scope_from_carried_list(carried)
    if carried is not None:
        raise refuse(f"step {step} does not take a carried list.")
    return ordered(frozen.checkpoint_ids)


def resolve_step_checkpoint(
    step: str, checkpoint_id: str, *, carried: Mapping[str, Any] | None = None
) -> SweepCheckpoint:
    """The one checkpoint a process drives or scores: allow-listed and in its step's scope."""
    record = resolve_checkpoint(checkpoint_id)
    scope = {item.checkpoint_id for item in step_scope(step, carried=carried)}
    if checkpoint_id not in scope:
        if step_record(step).narrowed_at_run_time:
            kind = "candidate" if record.role == "candidate" else "selected checkpoint"
            raise refuse(
                f"{kind} {checkpoint_id} is not justified by the carried list (a carried "
                "candidate, or the comparator of one)."
            )
        raise refuse(f"{checkpoint_id} is not on step {step}'s list.")
    return record


def allowlist_record() -> dict[str, Any]:
    """Return the whole allow-list as JSON-safe data, for manifests and checks."""
    return {
        "schema_version": ALLOWLIST_SCHEMA_VERSION,
        "checkpoints": [record.to_dict() for record in ALLOWED_CHECKPOINTS],
        "refused_update_zero_duplicates": [
            {"source_run_id": run_id, "checkpoint_relative_path": relative, "sha256": digest}
            for run_id, relative, digest in REFUSED_UPDATE_ZERO_DUPLICATES
        ],
        "suites": {key: suite.to_dict() for key, suite in SUITES.items()},
        "steps": {key: step.to_dict() for key, step in STEPS.items()},
        "run_id_pattern": RUN_ID_RE.pattern,
        "smoke_checkpoint_id": SMOKE_CHECKPOINT_ID,
        "smoke_stop_after_completed_episodes": SMOKE_STOP_AFTER_COMPLETED_EPISODES,
        "gpu_device_id": GPU_DEVICE_ID,
    }


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))

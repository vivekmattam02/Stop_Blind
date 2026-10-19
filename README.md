# StopBlind

Code, preregistration and run records for

> Vivek Mattam. **What Teacher-Forced Loss Misses: Stopping at the Goal in Fine-Tuned
> Vision–Language Navigation Policies.** 2nd Workshop on Embodied Spatial Reasoning, NeurIPS 2026
> (non-archival).

The camera-ready PDF is `paper/main.pdf`, and its LaTeX source is in `paper/`.

## The study

We fine-tune Qwen3-VL-8B-Instruct on Room-to-Room in Continuous Environments (R2R-CE), with and
without class weighting, three seeds each, and keep the checkpoint with the lowest teacher-forced
validation loss. In the Habitat simulator, on 61 `val_seen` episodes, none of the selected
checkpoints reaches the goal and stops there, and the selected checkpoints trained without class
weighting never stop. A preregistered follow-up executed all 37 distinct saved checkpoints on the same episodes, and
no checkpoint was shown to navigate better than the selected one.

The validation loss gives no warning because it is an average over decisions in which `STOP` is
rare. Split by the action the expert took, the loss of the selected unweighted checkpoint (seed
20260826, update 160; Figure 2 of the paper) is:

| expert action | decisions | mean CE | contribution | share |
|---|---|---|---|---|
| move forward | 173 | 0.397 | 0.2681 | 36.4% |
| turn left | 42 | 1.299 | 0.2131 | 28.9% |
| turn right | 38 | 1.395 | 0.2071 | 28.1% |
| **stop** | **3** | **4.180** | **0.0490** | **6.6%** |

The four contributions sum to the selection loss of 0.7373. `STOP` has the worst cross-entropy of
any action and is never predicted, yet it moves 6.6% of the number that picks the checkpoint.

## Contents

```
paper/                      camera-ready LaTeX source, figures, tables and compiled PDF
docs/PREREG_31_CHECKPOINTS.md   preregistration of the 37-checkpoint follow-up (see below)
src/spatialvlm/             the library: backbone, data, evaluation, training
scripts/                    training, evaluation, sweep, analysis, figure and table scripts
tests/                      unit tests
artifacts/spark/            result, manifest and verification records of the training runs and the
                            first closed-loop evaluation
artifacts/context/          derived analyses cited by the paper
artifacts/torch/            the follow-up: its preregistered analysis of all 37 checkpoints
                            (analysis_v1/) and the test run made before the freeze (stopblind_smoke/)
MANIFEST.sha256             SHA-256 of every file in this release
```

## The preregistered follow-up

The protocol in `docs/PREREG_31_CHECKPOINTS.md` was frozen in version control on 9 October 2026,
before any unselected checkpoint was executed or scored. It was not filed with
a public registry. The analysis output records the protocol's SHA-256
(`inputs.preregistration_sha256` in `artifacts/torch/analysis_v1/analysis_main.json`).

The follow-up ran on NVIDIA L40S GPUs (driver 610.43.02) in an Apptainer image of NVIDIA's PyTorch
25.10 container (torch 2.9.0a0, transformers 4.57.6, peft 0.18.1, habitat-sim 0.3.3). Its run records list the
SHA-256 of the 68 code files it ran, and the copies of those files in `src/` and `scripts/` are
byte-identical to them.

`analysis_main.json` holds, for every one of the 37 checkpoints, the offline criteria, the
closed-loop outcomes and the preregistered tests. Table 4 of the paper is generated from it:

```bash
python3 scripts/generate_stopblind_sweep_assets.py --out /tmp/table4.tex
```

Apart from LaTeX comments, the output matches `paper/tables/native_c0_sweep.tex`.

## Checking numbers

```bash
python3 scripts/check_paper_claims.py
```

This checks two values in the paper against the artifacts here and confirms that every generated
macro the paper uses is defined. The full claim check used for the paper also reads the per-episode
closed-loop traces, which are not included (see below).

## Redactions

Three files differ from the originals by an internal name or a local path. Nothing else in them is
changed.

| file | change | SHA-256 of the original |
|---|---|---|
| `docs/PREREG_31_CHECKPOINTS.md` | an internal reviewer's name; a local folder path | `0419b7bd6022056ae469077254aa2d620d560502035e0de13cfb853bd83c74e4` |
| `artifacts/torch/analysis_v1/analysis_main.json` | an internal reviewer's name in one listed file name | `e9561497e7583e9e7fdb2a61532248e6d89db9e15d1851814899ac08df10e161` |
| `scripts/analyze_native_c0_checkpoint_sweep.py` | an internal reviewer's name in one comment | `39871d591ec8031b8401d05619dd962393d3fccc3d3764108be7613727931949` |

Code comments and the protocol refer to internal review entries (`CR-0xx`) and to documents that
are not part of this release.

## Not included

- **Model checkpoints and cached training inputs.** They are derived from Matterport3D scans and are
  not distributed here.
- **Per-episode closed-loop traces**: about 1 GB for the first evaluation and 5.5 GB for the
  follow-up. The aggregate results and manifests are included. The traces are available from the
  author on request.
- **The scripts that submitted the follow-up to the cluster.** The run records name the GPU, its
  driver and the container image's SHA-256.
- **A second, unpublished study** that shares this codebase. Its runs, scripts and manuscript are
  left out, and nothing here depends on them.

## Environment

Python 3.10 or later; see `pyproject.toml`. Closed-loop evaluation also needs Habitat-Sim, the
Matterport3D scans and the R2R-CE episodes, each obtained from its own distributor under its own
terms. The launch scripts read their data and cache locations from environment variables and
default to folders under `$HOME`.

## Citation

```bibtex
@inproceedings{mattam2026stopblind,
  title     = {What Teacher-Forced Loss Misses: Stopping at the Goal in Fine-Tuned
               Vision--Language Navigation Policies},
  author    = {Mattam, Vivek},
  booktitle = {2nd Workshop on Embodied Spatial Reasoning, NeurIPS},
  year      = {2026},
  note      = {Non-archival}
}
```

## Licence

MIT; see `LICENSE`. The one exception is `paper/neurips_2026.sty`, the NeurIPS style file,
which is included only so that the paper compiles.

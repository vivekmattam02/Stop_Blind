# Preregistration: driving StopBlind's 31 unselected checkpoints

Frozen: the commit that adds this line is the freeze commit.

Ninth draft, 2026-10-08, written on the CPU; amended 2026-10-09 before any smoke. The fourth to seventh drafts are kept byte for byte
in `docs/prereg_history/` (the fourth with its addendum), and the eighth is in git at `0a32cff`.
The fifth draft was a lean version of the fourth: the scientific rules were kept, and the fourth
draft's own tamper machinery (an append-only addendum with a hash chain, an entry writer, a stamp
log and gates between steps) was replaced by git. The sixth draft kept that and restored H2:
every one of the 37 policies is scored offline on the study GPU type, as the first cluster job
after the freeze commit, and the analysis script computes the picks of rules B, C and G from those
scores. The seventh draft and the eighth change no hypothesis, test or bar of
sections 7 to 9. They answer a review of the code and the cluster kit, and its second reading:
which commits may come between the smoke and the freeze and after it (Freeze), how failed attempts
are counted (section 5), GPU use as a requirement of the smoke, two tighter scorer bars in the
smoke that decide only whether H2 runs, and what is done if the smoke fails (section 6). The ninth
draft changes tests and rules of sections 4 to 9, before any of the 31 has been driven or scored,
after an outside reading of the eighth (`reports/StopBlind Torch sweep audit.md`, its twelve
changes): among them the H1 screen (7.3), update 0's comparators (7.2), a sensitivity for successes
ended by an exact-tie STOP (7.3), O2's exact p-values (8) and the titles of rows 3 and 4 (9). On the
side of the smoke and the kit it adds a check of the simulator against the Spark's control runs
(section 6, phase 4), runs the six comparators first with more retries (5), gives the smoke and the
sweep one job size (6), and starts nothing on a node with another GPU driver (4.2). Earlier drafts
and the checks that shaped them are listed in `PREREG_revision_log.md`, which lists each change of
the ninth draft. A second reader read the ninth draft three times on 2026-10-08. The first reading's
findings changed what a missing episode does to the screen's family, to m and to the carried list
(5, 7.3, 7.5, 9), the tie-STOP sensitivity (7.3), the error-rate claim (7.3, 7.7), the order and
the pass and failure rules of the smoke (4.3, 6, 11) and when a run moves (4.2). The second
reading's findings, A to G, were fixed afterwards, among them what a missing episode does to the
rows and titles of section 9 and to the wording of 7.8. The third reading found no fault in any
test, bar, row or title; its five wording findings were applied before this draft was committed.
The revision log lists each change. On 2026-10-09, before any smoke, the author answered three of
the items this file marked for the author, and a unit was corrected: one GPU's share of an L40S
node is 128,250 MB, below the 128 GB (131,072 MB) each one-GPU job asked for, so on the L40S each
job now asks for 64 GB, which the author accepted, and the cap on P is the number of copies of the
model's weights the GPU can hold (section 6); section 6 gains a remedy for a smoke that runs out of
host memory; the rules for a driver move (4.2) and for a phase-4 miss (6) stand as drafted. No
hypothesis, test, bar, row or title changed. Later that day a second reader read this file and the code for
CR-054 after the first Torch smoke (job 19494919, which passed). Its findings changed two things
here: section 7.7 states that the error bounds treat episodes as independent and reports the
scene-level test on the confirmation suite too; and the analysis now admits an episode report only
as its own episode with its own step and timing files (section 5's completed report). The smoke
therefore runs again from the new commit. No test, bar, row or title changed.

## Freeze

The freeze is a git commit. Before the first job after the smoke, the author commits this file,
the runner, the offline scorer, the analysis script, the confirmation-suite builder and its output,
the smoke's scene check with the modules of the Spark's control suite that it runs (among them
`src/spatialvlm/eval/policy_ordering.py` and `src/spatialvlm/eval/habitat_suite.py`), the
eight-episode smoke list, the list of model snapshot files with their sha256, the committed
smoke result, and the cluster kit (section 4.3). Just before that commit, the line "DRAFT. Not
frozen..." at the top is replaced by "Frozen: the commit that adds this line is the freeze commit."
The offline step and the main run (section 5) run from the freeze commit, or from a later commit
that changes only what may change after the freeze (below). Everything from the smoke's commit to
the analysis lives on one git branch, the study branch, which holds no other project work.

- **What every result records.** Every manifest and every episode report records the commit it ran
  from and this file's sha256. The cluster receives the code as `git archive` of that commit, so the
  code that ran is the commit's. The one addition is `third_party/pydeps`, which git does not track
  and which only the image build reads; the bundle names it.
- **What may change after freeze.** Commits after the freeze commit may add result files (under
  `artifacts/torch/`); may change any file of the cluster kit's folder `scripts/hpc/stopblind_sweep/`
  (its shell scripts, its host-side Python, which no job imports, its self-test and its runbook);
  and may change Markdown files other than this one, such as `CURRENT_STATE.md`,
  `docs/CROSS_REVIEW.md` and the results ledger. They may not change this file, any other Python
  file, a test, a configuration file or anything else. Between the smoke's commit and the freeze
  commit the same holds, and this file may in addition change its status line, and nothing else in
  it. The analysis script checks all of this with `git diff --no-renames --name-only` between the
  freeze commit and HEAD, and between the freeze commit and every commit a job ran from; it refuses
  anything else, and lists every later commit a job ran from in its output. Every record also lists
  the sha256 of each Python file the job loaded, and the analysis script checks each against
  `git show <commit>:<path>` for the commit the record names. The same checks, from the same
  module, run in the bundle builder before it builds a bundle of a frozen commit, so a bundle whose
  results the analysis would refuse is never built.
- **Results are committed as they come back.** Each run's directory comes back from the cluster as
  one tar; its sha256 is printed on the cluster and computed again on the workstation, the two must
  be equal, and the directory is committed to the study branch before any analysis reads it.
- **Before the freeze commit,** a reader who did not write this file checks every number and rule in
  it, a reader who did not write the code reviews the code, and the project's cross-review entry for
  both is closed by someone who wrote neither. CR-048 is settled first: its index row is narrowed to
  the rescue runs, or the author's verdict on it is recorded.

**Changes after freeze.** Sections 5 to 10 and the rules of section 8 do not change after the freeze
commit. Any other change is reported as exploratory, beside the preregistered result and never in
its place. The author may push the freeze commit, as a tag, to a hosted remote of the author's own,
which records when the push arrived; the author cannot set that time, as the author can set a
commit's own date. The result commits stay off the project's shared remote: the main run's results
take about 6 GB, GitHub, which hosts that remote, refuses a push of more than 2 GB, and the remote
belongs to a collaborator. Whether to push the tag is the author's decision **[!]**.

Evidence order used here: artifacts, then code, then `CURRENT_STATE.md`, then the ledger. No number
comes from a `*.invalid_cr048_*` directory.

## 1. The question

Each of StopBlind's six training runs saved seven checkpoints, at updates 0, 32, 64, 96, 128, 160
and 192. The preregistered rule picked one per run: lowest teacher-forced cross-entropy (CE) on 256
held-out natural-mix decisions, with exact ties going to the earlier update. Those six were driven
through 61 episodes on the DGX Spark on 2026-08-26, and each succeeded in 0 of 61. The other 36 were
never driven.

Four people at two venues raised the same gap: reviewer 7BzX and area chair FsH9 at VLM4RWD, and
reviewers zx7k and jsSy at ESR. zx7k: "evaluating the saved checkpoints would quantify the
selection-specific effect". jsSy: "there may have been no good checkpoint to pick in the first
place". This experiment drives the unselected checkpoints on the same 61 episodes on NYU's Torch
cluster, and drives the six selected ones again beside them on the same GPU type, so that every
comparison is made on one type of hardware. It fixes now, before any of them runs, what each
possible result will mean.

## 2. What is known before any new run

- **The files.** All 42 checkpoint files are on disk, and all 42 match the sha256 recorded in their
  run's `result.json` (re-hashed by two independent checks on 2026-09-30). The six `result.json`
  files hash as `native_c0_closed_loop_v1` pinned them.
- **The six selected checkpoints on the Spark** (`native_c0_closed_loop_v1`, one NVIDIA GB10):
  success 0 of 61 each (exact two-sided 95% upper bound 0.059); oracle success 1 of 61 each; the
  unweighted ones match the always-forward control on 61, 61 and 60 of 61 episodes and never press
  STOP; the weighted ones press STOP 7 times in all, each 6.63-12.80 m from the goal. 53 of their
  closed-loop decisions were exact BF16 ties, and in wt25-u192 49 more were won by one BF16 step
  (0.125 nats). Another GPU can break such decisions the other way.
- **Speed on the GB10.** A median of 0.221-0.229 s per decision; 28,023 to 30,500 decisions per
  checkpoint (61 x 500 = 30,500 when it never stops); 1.93-2.07 h per checkpoint with overhead.
- **STOP on the 256 selection decisions.** All 18 trained unweighted checkpoints predict STOP 0
  times. Update 0 predicts it 28 times. The 18 trained weighted checkpoints predict it 0 to 34
  times.
- **The existence gate** was computed at the six selected checkpoints only, and failed at all six.
- **The suite.** 61 `val_seen` episodes in 53 scenes, one per physical trajectory; the shortest
  runtime start distance is 6.598 m. Expert replay succeeds on 61 of 61; always-forward and
  immediate STOP on none.
- **Nothing on Torch.** No GridNav job has run there yet.

## 3. The checkpoints

**Runs.** Each tag names the treatment and the seed. `uw` is unweighted; `wt` is inverse-frequency
class weighting, with STOP weighted 28.507 times FORWARD (2024/71, the same in all three weighted
runs).

| tag | `source_run_id` | gate | run manifest sha256 | selected update |
|---|---|---|---|---|
| uw25 | `native_c0_development_v1` | `native_qwen_c0_development_v1` | `726952465b43827384ecb6bc20a41d867fbb83a3027c4cae30ebe00ccb38ab7b` | 160 |
| wt25 | `native_c0_action_balance_v1` | `native_qwen_c0_action_balance_v1` | `056284189286c66de34fbe0dd5b4b9f59ecc1d805688a49440692cfa5548867a` | 192 |
| uw26 | `native_c0_seed_replication_unweighted_s20260826_v1` | `native_qwen_c0_development_v1` | `a6a2971b11aea148fa2158b02166f88bce1c5845a6496013e89bb0b61d8fdfb0` | 160 |
| wt26 | `native_c0_seed_replication_inverse_frequency_s20260826_v1` | `native_qwen_c0_action_balance_v1` | `d78f32d8daa0548cdbe0feaacf2577b6dd488110119fcd648a248a74a1ccebee` | 128 |
| uw27 | `native_c0_seed_replication_unweighted_s20260827_v1` | `native_qwen_c0_development_v1` | `be24ec75a8913e7b527a84ecfb43d5e6aff143e52aa87f1b2dd7a4fc66d9117e` | 160 |
| wt27 | `native_c0_seed_replication_inverse_frequency_s20260827_v1` | `native_qwen_c0_action_balance_v1` | `61312cc64871b81b0fc51f20681f85636b04b223539a60253bf5d1f6a47385fd` | 192 |

The run order above is the closed-loop runner's own order. Run artifacts are in
`artifacts/spark/<source_run_id>/`. Files are under a local checkpoint folder.
"Selection CE" and "STOP predicted" are the values recorded at training time
(`calibrations[k].summary`; STOP is counted as `predicted_action == "STOP"`, so an exact tie counts
as STOP).

**All 42 checkpoints.**

| # | id | file | sha256 | trainable_state_sha256 | selection CE | STOP predicted /256 | role |
|---|---|---|---|---|---|---|---|
| 1 | uw25-u0 | `native_c0_development_v1/step_0000.pt` | `2dcae1f8297c09d34b853892191078b0babd801809d2cce757ea0fd77e9e2c66` | `e7437fb6ee1f20411906d09b1c4ff072b905c1578798ad0ac2ebb314b753aa25` | 5.4872 | 28 | **candidate** (the one update-0 copy) |
| 2 | uw25-u32 | `native_c0_development_v1/step_0032.pt` | `d75b372f5d68db92a2b870a283b8545bd3246b7316b7b88cf9d4c692d20d789c` | `f2fc80acd73992e03e9db6db107a6875fe594ce9a274267ab457e95d77429f37` | 0.7795 | 0 | **candidate** |
| 3 | uw25-u64 | `native_c0_development_v1/step_0064.pt` | `95853ea867f6802038153d6a78b229d817db2b95e67ba28088eb04ee8a22725c` | `059743c7bdd77c586f077c6523647c0f26bb08687f3f84b11f049a1e66100c93` | 0.8139 | 0 | **candidate** |
| 4 | uw25-u96 | `native_c0_development_v1/step_0096.pt` | `fd52bd1c84f9e180a29a1709424a1c25be348cd96af0cbc62dbcc94e9a4c0cd0` | `7d8b7b1cbc29d36b3d1ee61bd5c63a298e93efcc37b02ad4f490439dd77406fd` | 0.7732 | 0 | **candidate** |
| 5 | uw25-u128 | `native_c0_development_v1/step_0128.pt` | `f9ef8df099ee907e1f1b5a1005d5314ba2c61a444fae9ee3d5c67e9edcb8c9ad` | `5a0f18addd2e995f45baf3fa5f426140410df708cebef52f5eed9a9be62d8bfe` | 0.7573 | 0 | **candidate** |
| 6 | uw25-u160 | `native_c0_development_v1/step_0160.pt` | `a4a58cd34409e24c805c847249cd921da17d11c2bef96e37a20bf7c50a86987d` | `1bbb8490f51ca79c5109058a943cfa5c011de9ce69cfaf088dcdff672d5d984f` | 0.7342 | 0 | **selected** (driven on the Spark 2026-08-26) |
| 7 | uw25-u192 | `native_c0_development_v1/step_0192.pt` | `566b58d110e1196c1a94c1739df6bba518115d21d675726d05c2f8e4d570f68d` | `518bcc8f4f51fc4f96399d5b670f7634df3d05f1f79af7fe9773d13d6297c14c` | 0.7411 | 0 | **candidate** |
| 8 | wt25-u0 | `native_c0_action_balance_v1/step_0000.pt` | `7ea1269caf1ba2d7b5e1542c31d4a6819efa76b4d1330ba40528b016ccd38c1f` | `e7437fb6ee1f20411906d09b1c4ff072b905c1578798ad0ac2ebb314b753aa25` | 5.4872 | 28 | update-0 duplicate, not driven |
| 9 | wt25-u32 | `native_c0_action_balance_v1/step_0032.pt` | `e884bca4685fe6ebe4b3463cfeefa6c9e5976ebe84dad150a67a48c957ae5f9f` | `b4105798c599e205541ab5efcc37eea3b497ca37b051f4c89b6b9c1bd9e3de85` | 0.9836 | 1 | **candidate** |
| 10 | wt25-u64 | `native_c0_action_balance_v1/step_0064.pt` | `8aad706e03b626282f14418fe464412d0df2c898e7b42926ed54d84601b67236` | `f492d61eeff338ac624c2a287ad1a5eba9b1a4bf3370c0a75af21ac1968a94b0` | 1.2008 | 2 | **candidate** |
| 11 | wt25-u96 | `native_c0_action_balance_v1/step_0096.pt` | `27d036a50d1b5be4ef5a9cfcf09bfc5d36e44a20cce9da96aa25df3d7a5acf7d` | `486a1c6910b39a44e03d79fe9c20257083074d0eec930650b6d553543ddbdca7` | 0.9544 | 0 | **candidate** |
| 12 | wt25-u128 | `native_c0_action_balance_v1/step_0128.pt` | `717750ec416fd4f67281cda0eefe38141a0c6ad31a13905eeb112971a7dae63c` | `a0958939a7db9297e06e03956279ef69b82df44af9fcbe26fe97479af75fdf80` | 1.0183 | 4 | **candidate** |
| 13 | wt25-u160 | `native_c0_action_balance_v1/step_0160.pt` | `7ef6df2cc477fded67cbd54337feec4b10f591424cd67a06ee9bd7de96e21fb0` | `e2430c68ea2bdc881a889e4f3de706d3a94e6c02061bbbe4999a05b9453df976` | 0.9884 | 5 | **candidate** |
| 14 | wt25-u192 | `native_c0_action_balance_v1/step_0192.pt` | `f21496af3bc35bcc9f680b142bb19b4eff7f92fb5e8b405837e50ff7181872dc` | `67381fcef28903f537e985a064a6ab9088dde0764f6f5fcd51bc10efb71cbbbc` | 0.9406 | 2 | **selected** (driven on the Spark 2026-08-26) |
| 15 | uw26-u0 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0000.pt` | `d361ac03db673243fc1c66b056c40b4d1717087f8cf0adca54d01ca0bb4d38c1` | `54de1d7ebbb41e11c2f6a51c838fc3c0fb38ee914bfb7d22889cc83637bded2e` | 5.4872 | 28 | update-0 duplicate, not driven |
| 16 | uw26-u32 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0032.pt` | `ecc23c53ec8756053f36938c881d60e47a8e329ef89d480566106ed49e075886` | `73ebeb0fe42bc55ca31313c0d629f6886ec019836abe237c4ab74b724420fa38` | 0.8101 | 0 | **candidate** |
| 17 | uw26-u64 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0064.pt` | `c259b21a148bcfa21caa0218a9964e344571b79b9574914dc74ca19366cda280` | `d7af475b1f353fb6f5e49b580788ced9c9034f2f0c25db56d1da1770cf3e19c1` | 0.7977 | 0 | **candidate** |
| 18 | uw26-u96 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0096.pt` | `a01ec71931282de2ba8c3422ff2b7e994d3efc130b49ee52c1cd6eee6a5af4f4` | `f8bab93e16d7d817b3e4058bf02e8166562bde4675a3ba048b94f382744d4c54` | 0.7594 | 0 | **candidate** |
| 19 | uw26-u128 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0128.pt` | `53cff620c3a2aa677d9f5afde78c1a2bb4fade241f4c87375339d281d3b08228` | `1f4a7deb612b96e01d504d78f6a42545bc9d4d3e2ed6ce7e523e088d97a13715` | 0.7604 | 0 | **candidate** |
| 20 | uw26-u160 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0160.pt` | `ac5b106093c3b7e0a30851f907bba094b0db35f382940394bbc6d95fd4b89413` | `f9e43568acfe84bdef1ae311d3bed6beb7f47160d1ae9f5d43b0c4840a265c53` | 0.7373 | 0 | **selected** (driven on the Spark 2026-08-26) |
| 21 | uw26-u192 | `native_c0_seed_replication_unweighted_s20260826_v1/step_0192.pt` | `247c203c04f2ddc266dbb7fbbf5122f6d3f271938d9f0b3437d631307457645b` | `950eeee6a18804e87fc3ae4bb61502a510df7138a9ea803206cd11d94ab3e0e5` | 0.7427 | 0 | **candidate** |
| 22 | wt26-u0 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0000.pt` | `0e3546adfee4d2dca3dff789345eca1c07c696c97522851d1f5c362d3b078861` | `54de1d7ebbb41e11c2f6a51c838fc3c0fb38ee914bfb7d22889cc83637bded2e` | 5.4872 | 28 | update-0 duplicate, not driven |
| 23 | wt26-u32 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0032.pt` | `6f3c44e8aa57358e766007c1912a6b8cf305a2c6ad02b14de52fb62642fe4b20` | `0f5832372d7dce4d2bdedec32c8b4f642430bc49301715b8ec0cb45d80c48b1f` | 1.3408 | 7 | **candidate** |
| 24 | wt26-u64 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0064.pt` | `69e2be613e1708295c85f5c7afb3e4452cdddcd5362d5b6bd9be6f81640d1f30` | `fd3ba132bde955f7bc2b7abd57d2305afc2a0565c57bb18ab0c36f0a609d9b0b` | 1.1128 | 34 | **candidate** |
| 25 | wt26-u96 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0096.pt` | `84ecf502c1a24f8f52678ed054a5b65ad8143695712b725dc6be132b1f43d4e3` | `8f8d2966c77f2ca19f2dbd7a6d382fff9f99a2700ea542dddf13df5bb0fce991` | 1.0119 | 0 | **candidate** |
| 26 | wt26-u128 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0128.pt` | `55380852648df9581c06b887f5399240f981245d4da7b868019fb6ea8a544260` | `71f9a4bdc688b3bf2914d9c7f584d32e57f593d0f8e7f509b6a895c06a3c80c8` | 0.9389 | 0 | **selected** (driven on the Spark 2026-08-26) |
| 27 | wt26-u160 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0160.pt` | `18626bea33bb083a7b11a140cfdd5223e108701589f6e956484c737c6a9cd8a0` | `dac2553daff6880ceacb362203ca2049363c098e4ac41078c297df4eadc91543` | 1.1567 | 4 | **candidate** |
| 28 | wt26-u192 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/step_0192.pt` | `f04d62da03568bf46e586189a310ccdcdf14aabcb595e265d1eca370649e5f98` | `3264315d412e330185ce054885f28a26d8ae74abbc3ef1ef0f0d2964785e37bd` | 0.9418 | 9 | **candidate** |
| 29 | uw27-u0 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0000.pt` | `1643dd842258667e006990ca6e4cae5294d3793bbe947dd7e197b92fa6203132` | `71c61baa4573b1d6031b066d3b2e8570aaa3f38cf9b9f70e09a693863d8b3f71` | 5.4872 | 28 | update-0 duplicate, not driven |
| 30 | uw27-u32 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0032.pt` | `315ace9c9cf3ddf43e21d8d28385dc0ae1efd941b060661104f025f227928a52` | `008e8699736f549f900e58a2662660cce7dec74a32092c3267c1fd1275c5fcbd` | 0.7964 | 0 | **candidate** |
| 31 | uw27-u64 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0064.pt` | `84257d577bac7aa5265a04b8a9cf9c6e4af662befa774979c2cad68b0be07b1a` | `3654a1d6cc5d460825c625a9ded1dab46503be539a6883e7d22e4706c7fc5ae5` | 0.7772 | 0 | **candidate** |
| 32 | uw27-u96 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0096.pt` | `fb935b2d21b90cf19afc4e7ccc4962847d2763f185f9218964ec98c4f0cb0e5e` | `2ffaf19251683bd47714b42e14c678a9e99a9f77595a1b72cb6fa9794c0af224` | 0.7662 | 0 | **candidate** |
| 33 | uw27-u128 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0128.pt` | `5ad08189aad45d6a2bd4d712a2a4658a1d0221f791937ed1e63dc501517fcad4` | `03d49dd55527f6450a42e0c0dd9e55426f67c226013790997be01a0da83f9292` | 0.7626 | 0 | **candidate** |
| 34 | uw27-u160 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0160.pt` | `7264fdd47d9554ce9d21da8f29d7ccb73dbe3903560b587c26106297cf86da21` | `9e56e84b11eeecacb6c8b8186a0e74e4d3d99f39933be61baa6b826a00a83fba` | 0.7478 | 0 | **selected** (driven on the Spark 2026-08-26) |
| 35 | uw27-u192 | `native_c0_seed_replication_unweighted_s20260827_v1/step_0192.pt` | `e344473921433c1e7a3f34abbb3894733ac0f162394fb237dc94b1af95756aa3` | `8f354bd0932174dfbeb573c20c98e098a8b338604ff155c1651a77c518d23d52` | 0.8097 | 0 | **candidate** |
| 36 | wt27-u0 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0000.pt` | `54004b26549e2ccea838fa7230dc2de2473fa3f48b111ff7060043115643f53d` | `71c61baa4573b1d6031b066d3b2e8570aaa3f38cf9b9f70e09a693863d8b3f71` | 5.4872 | 28 | update-0 duplicate, not driven |
| 37 | wt27-u32 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0032.pt` | `119ddddda95f6375878373f6b00cda6db88bd97009d1f4460ca6e19cd449e0f7` | `ed499aafb2ed2da3a61a2a979569fa8fb1511bd7bdc60587e84627b1f3a0565e` | 1.0656 | 3 | **candidate** |
| 38 | wt27-u64 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0064.pt` | `c7f9788a1d10d0471c75327ff0d2cff9a7c3992c3685adc0de8472a1f1c38c3b` | `27821afcfb804eeebe66a89dce953feda59fe521267cca9521a47ee4cc4bef53` | 0.9801 | 0 | **candidate** |
| 39 | wt27-u96 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0096.pt` | `e4c1d0fc52aeb69dd98f72952436c1ba3d349fecc087d983eac63154fa1ece17` | `fbbcd331ee8330af693c3bdcedab7c557b3170196e6b6bc187a0db715d2ea70a` | 1.0076 | 1 | **candidate** |
| 40 | wt27-u128 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0128.pt` | `f9e1418ab8bfdff4ae83402f4c96754227f04431feee28242b26574de623db63` | `bb0fd4501708582cd3f0f4c09295e4073b1779c888daaba2cb0ec723ff518928` | 1.0528 | 0 | **candidate** |
| 41 | wt27-u160 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0160.pt` | `ca3751c1bcfd5908f0914e3a48d866d364a63f6db2d8219b1589d86e9905aa71` | `3ce46b906e29b2e104f33fabf7fb034976e37bcf62d86458570ddedf66281450` | 1.0831 | 23 | **candidate** |
| 42 | wt27-u192 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/step_0192.pt` | `77f6eb8695323db53cf7394125f33a4a78be0b9608c500dfe575b4078eec81fc` | `36c807329351d3c60882a4728e7edfec7830c7cb57f440d32f2b065854b3639d` | 0.9230 | 1 | **selected** (driven on the Spark 2026-08-26) |

**Count.** There are 30 trained unselected checkpoints and one update-0 policy: these 31 are the
candidates. With the six selected checkpoints, which serve as their comparators, 37 distinct
policies are in play, and the main run drives all 37 on the same 61 episodes. These 37 are the rows
marked **candidate** or **selected**, and they are exactly the checkpoints the runner extension
accepts (section 4.3). The five rows marked "update-0 duplicate" are refused.

**Why the five other update-0 files are not driven.** The six update-0 files differ in bytes and
hold three distinct stored states, one per seed: `e7437fb6...`, `54de1d7e...` and `71c61baa...`.
They are still one function, for two reasons:
- in all six files, all 144 `lora_B` tensors are exactly zero (read from the archives on the CPU on
  2026-09-30, twice, by two readers). The adapter therefore adds exactly zero, and each file is the
  frozen base model. In each of the 36 trained files, no `lora_B` tensor is zero;
- their scores on all 256 x 4 selection entries are equal as floats in all six runs.

uw25-u0 (`native_c0_development_v1/step_0000.pt`) is driven, and its result stands for all six.

## 4. Apparatus

### 4.1 Held fixed from `native_c0_closed_loop_v1`

| item | value |
|---|---|
| model | `Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, a local snapshot of 14 files, offline; rank-32 LoRA (alpha 64, dropout 0) on q, k, v and o. The base runs in BF16; the adapter is held in FP32 (all 288 saved trainable tensors are FP32) |
| policy | H4 causal history (four frames with the previous actions; the first observation left-padded); one action per decision, chosen greedily by the NLL of the four action tokens; an exact tie goes to the lowest action id: STOP 0, FORWARD 1, LEFT 2, RIGHT 3 |
| task rules | 500-step cap. Success means the policy chose STOP and the final geodesic distance is below 3.0 m. Forward is 0.25 m, a turn 15 degrees |
| simulator | RGB 224 x 224, depth 256 x 256, field of view 90 degrees, sensor height 1.25 m, agent radius 0.1 m, height 1.5 m, sliding on, seed 100 |
| episodes | the 61 entries of `artifacts/spark/native_c0_closed_loop_v1/manifest.json` (sha256 `9d8e272d194106b924d4eee7e46e7dc3e478e023020954283773a788368a269a`), in the same order; they equal the control suite `artifacts/spark/policy_ordering_v1/suite_manifest.json` (`cb75d6259e42decccc161d069df2e04fdadbb7b4f20c6c480bcbca575ec44af4`) entry for entry |
| release | `val_seen.json.gz` `e52a19e08a2f6f936f61b021cf4a0e17ddb8685f8807c60ba6382a61e2463df0`; `val_seen_gt.json.gz` `cf41ac4dc6bcbada202e9cf0c414b0469153a4e1c5d02a21f088edd6032299fc` |
| code | the episode loop, policy, scorer, loader and environment of the locked runner, called unchanged. The 14 files that `native_c0_closed_loop_v1` locked must hash as that run's manifest records them (`software_sha256`); the runner checks this at start |

The locked runner cannot drive an unselected checkpoint, so the extension drives each of the 37
through the run's `checkpoints[]` entry, with the same loader arguments the locked runner passes
and the same per-episode report, plus `legacy_checkpoint_id` for the six selected checkpoints.
After every load it asserts that all 288 trainable tensors are FP32. CR-048, still `OPEN`, concerns
the rescue closed loops, which resolved a `val_seen` suite against the `train` release; the runner
binds every suite to the `val_seen` release before driving it.

### 4.2 One GPU type per run

All rollouts of a run, the six comparators and any confirmation drives included, and every offline
score, run on one GPU type of the cluster: the study GPU. The author named it on 2026-10-07:
**L40S** (`--constraint=l40s`), from the types the cluster offers (H200, H100, A100, L40S, RTX Pro
6000, or B200 if reachable). A move to another type follows the rule below. The smoke records its
Slurm constraint, the exact name `nvidia-smi` reports and the GPU driver. A job that lands on a
node whose GPU reports another name (another variant of the type) starts nothing and records the
node. So does a job that lands on a node whose GPU driver differs from the committed smoke
result's. If job starts on three distinct nodes are refused for the driver, counted across
the run's offline, main and confirmation steps, no further job is submitted and the run moves
(below); on 2026-10-09 the author left this rule to the drafters, who kept it. The trigger is that count of nodes alone. Every manifest
and every episode report records the GPU's name, UUID and driver, the node, and the image's
sha256. The runner refuses a GPU whose name differs from the one
its run's manifest records, and the analysis script refuses any set of results that holds more than
one GPU name, more than one GPU driver or more than one image. The driver counts as much as the
name: the host's driver supplies both libcuda and the EGL renderer that draws the agent's pixels.

The reason is the 53 exact ties and the 49 one-step wins of section 2: one type of hardware can
break them differently from another, so a candidate and its comparator must run on the same type.

**If a run must move.** If the study GPU becomes unusable (it leaves the cluster, the account loses
it, or no job of that type starts for 14 days), the run moves whole: a smoke on the new type, then
the offline step and all 37 checkpoints again on the 61 main episodes under new run ids, then
confirmation if it applies. The stopped run is reported as an aborted run, with all its counts,
beside the new one, and is never pooled with it. After the driver refusals above, the run moves
the same way, to the same type with the new driver or to another type: a new smoke there, then the
offline step and all 37 checkpoints again under new run ids. Neither a move's trigger nor the
choice of the new driver or type uses any outcome of the run. A move is never made to obtain
another result, and the paper reports every move.

### 4.3 The image, the kit and the code

**The image.** An Apptainer image built from `scripts/hpc/container/gridnav_torch.def` (sha256
`bd3862118ecfc43db25185a2511556bba929749062bf425ac054fd7b30453003` on 2026-10-04), on its primary
base, NGC PyTorch 25.10 for amd64. Its own test compares every Python distribution with
`scripts/hpc/container/versions.lock` and must report 0 errors: Python 3.12.3, torch
`2.9.0a0+145a3a7bda.nv25.10`, transformers 4.57.6, peft 0.18.1, flash-attn `2.7.4.post1+25.10`,
numpy 1.26.4, pillow 10.4.0, habitat-sim 0.3.3 at commit `acbe6f49`, the Spark image's versions.
The declared differences are five Intel packages, the GPU architectures torch was compiled for,
and Ubuntu packages fetched on the build day; every compiled binary differs, because the processor
architecture does. The fallback image of the same folder (torch 2.8) is not allowed. The job script
computes the image's sha256 on the host and passes it to the runner.

**The model snapshot.** The same bytes as the Spark's 14 snapshot files: downloaded from Hugging
Face at the recorded revision and checked file by file against the committed list of their sha256
values, or, if that check fails, copied from the Spark as one tar so that the cache's links
survive. Either way the runner checks each file against the committed list once per job: the job's
first act hashes all 14 files with the runner's own code, and every process of that job checks that
record and each file's size and modification time, so that 17.5 GB is not read again by every
process.

**The content cache.** The offline step reads the 1,152 held-out decisions of the C0 content cache
(256 selection, 640 natural and 256 balanced; 0.70 GB), moved as one tar; each object is checked
against the committed cache index before the step and again as it is read.

**What the code must do,** and nothing more:
1. **Drive and score.** The runner: `--step smoke|main|confirmation`. Main drives all 37 checkpoints
   of section 3 on the 61 main episodes; confirmation drives the carried list of 7.5 on the
   confirmation suite; the smoke drives wt25-u192 on the eight episodes of section 6. The offline
   scorer: `--step smoke|offline`. Offline scores all 37 on the three held-out surfaces of section 8
   with the training function itself; the smoke scores wt25-u192. Both refuse any checkpoint not
   among the 37 or whose file sha256 differs from section 3, and one process drives or scores one
   checkpoint. The smoke's scene check (`scripts/check_native_c0_sweep_scenes.py`) drives the two
   controls of section 6, phase 4, and loads no model.
2. **Record.** In the manifest, in every invocation record, in every episode report and in every
   offline record: the commit, this file's sha256, the GPU's type, name, UUID and driver, the node,
   the image's sha256 and the job id with Slurm's restart count, so that each job start has its
   own value. Refuse a GPU name or an image that differs from the committed smoke result's, and,
   after the smoke, a preregistration that is not frozen. Once per job start, the kit also records
   the numerics in force (TF32, reduced-precision reductions, the enabled SDPA backends, cuDNN)
   with the driver; a difference from the smoke's is reported, not refused.
3. **Resume per episode** (section 5): a completed episode report is the unit (offline: a scored
   policy); on SIGTERM, exit without a report for the episode in progress; count errors per episode
   and record an episode (a policy) as not completed after three that span at least two job starts.
4. **Measure in the smoke:** the scorer check of 7.4 and the smoke's two tighter bars on wt25-u192;
   peak GPU memory, seconds per decision and GPU use for one worker and for one scorer; the number of
   workers per GPU, P, by the rule of section 6; then P workers at once, and P scorers at once; the
   dtype check after every load; and the scene and simulator check of section 6 (phase 4).
5. **Run on the cluster:** one GPU per job, with the smoke's cores and host memory (section 6), P
   workers per job, one checkpoint per worker, the study GPU's `--constraint`, preemptible with
   requeue, a `singleton` dependency per job name; the code sent as `git archive` of the commit;
   the offline step before main; in every step the six selected checkpoints submitted first, with
   more retries (section 5). A job that sees a model load fail, or the renderer fail in an
   invocation's first episode, stops starting checkpoints and records its node, so that the next
   submission can exclude it (section 5); a job on a node whose GPU name or driver differs from the
   smoke's starts nothing and records the node (4.2). A program whose progress record has not
   changed for 30 minutes is stopped, as a failed attempt (section 5). The offline scorer updates
   its record at least every 100 decisions, and the smoke fails if a scorer or runner process of
   phases 0 to 3 went more than 15 minutes, half that limit, without an update, or if such a gap
   could not be measured (section 6).
6. **Analyse** from committed results only: the scorer check of 7.4, the criteria and the rule
   picks of section 8 from the offline records, the tests of section 7 with the tie-STOP
   sensitivity of 7.3, the missing-episode rule of section 5, the descriptions of section 8, the
   cross-check of 7.8, the carried list; refuse mixed GPU names, drivers or images, and commits
   whose difference from the freeze commit touches anything but what may change after the freeze
   (Freeze).
7. **Test on the CPU:** the screen of 7.3 (its bars for c = 0 to 4, and cases where it differs from
   Bonferroni over 31 and from Tarone's procedure at 0.05); update 0 against all six selected
   checkpoints; the bars of 7.4 and 7.5; the tie-STOP sensitivity, with the tie read from an
   episode report's last decision; the missing-episode rule; the refusal of mixed GPU names and
   drivers; O2's exact p and smallest attainable p against an independent enumeration; a resume
   after SIGTERM on a fake episode loop; the cross-check's wording rule on synthetic pairs; the rule
   picks of section 8 (A and D from the six `result.json` files, B, C and G from an offline table);
   the offline step on fakes that return the recorded decisions, and the scorer check on them; the
   commits allowed around the freeze; failed attempts counted over job starts; and the scene check's
   pass rule, with a fake simulator that replays the Spark's recorded traces: a port that
   reproduces them passes, a replay that ends 0.3 m from the Spark's final distance fails, and a
   confirmation episode that does not load fails.

The runner extension written for the third draft already drives the 37 through `checkpoints[]` and
resumes per episode. Its aarch64 check, addendum reader, step gates and verifier are removed; its
offline scorer is kept for the offline step.

## 5. Order of work, resume and missing episodes

| step | where | what |
|---|---|---|
| 1 | CPU | a reader who did not write this file checks it; the code of 4.3 is written, tested on the CPU and reviewed by a reader who did not write it; the confirmation suite is built (7.5); the cross-review entry is closed and CR-048 settled |
| 2 | cluster, at most about 3 GPU-hours, approved by the author **[!]** | the author names the study GPU, moves the data, the checkpoints and the snapshot, builds the image, and runs the smoke of section 6 from a commit of the study branch; its results come back and are committed. If the code changes after a smoke (anything but what may change after the freeze), the smoke runs again from the new commit |
| 3 | | the author approves the hours of section 11 **[!]** and makes the freeze commit, which differs from the smoke's commit only in this file's status line and in what may change after the freeze (Freeze) |
| 4 | cluster | offline, the first job: all 37 policies scored on the three held-out surfaces, P workers per GPU, on the study GPU's type; then main: all 37 checkpoints on the 61 main episodes, P workers per GPU; both from the freeze commit or from a later commit allowed by the Freeze section |
| 5 | workstation | the offline and main directories come back and are committed; the analysis script, from the study branch, computes the scorer check, the criteria and the rule picks of section 8, and the tests of section 7; a reader who did not write the script re-derives the results and the carried list |
| 6 | cluster, then workstation | confirmation, only if the carried list is not empty; its results come back, are committed and are analysed |

The author runs every command on the cluster and pastes the output back; the session that prepares
the code never logs in.

**No early stopping.** All 37 run to completion in both steps, whatever the interim results. A
worker stops only at its job's time limit, on an interruption by the cluster, or on an error. Nobody
stops a job by hand except for an infrastructure failure, and any such stop is noted in the run's
log with its reason.

**Resume, per episode.** The unit of resume is a completed episode report with its step and timing
files, as in the locked runner. A requeued job may start on any node of the study GPU's type; each
worker continues its checkpoint at the first episode without a report.
- **Interruptions** (preemption, the time limit, a node failure, a cancellation for low GPU use):
  Slurm sends SIGTERM; the runner exits at once without a report for the episode in progress, and
  that episode runs again from its first step, with the model reloaded and a new simulator. An
  interruption is not a failed attempt.
- **Errors:** an episode that fails with an error runs again. After three errors that span at least
  two job starts it is recorded as not completed; so is every remaining episode of a checkpoint
  whose model failed to load three times over at least two job starts, the dtype check included. A
  job start is one start of one Slurm job on one node, named by the job id with its restart count
  and by the node: a requeued job, a new submission or another node is a new start. The rule keeps
  one faulty GPU or file system from using up an episode's or a checkpoint's attempts within
  minutes. In the offline step the policy is the unit: after three errors over at least two job
  starts it is recorded as not scored. Within one job start the kit retries an error once, and an
  error of one of the six selected checkpoints three times (Comparators, below). After a failed
  load, and after an error of the renderer (EGL, Magnum or the windowless GL context) before an
  invocation has completed its first episode, the kit does not retry the error on that node: it
  starts no other checkpoint there, records the node, and ends the job, so that the next submission
  can exclude the node. The error still counts, as a failed load or as a failed attempt of the
  episode in progress. The same renderer error later in an invocation is an ordinary error and is
  retried. A program whose progress record has not changed for 30 minutes is stopped by the kit;
  that stop counts as an error, a failed attempt, and not as an interruption, so a program that
  hangs cannot be retried without end.

**Missing episodes.** A not-completed episode is never dropped, every bar stays at 61, and it is
listed by name; a checkpoint whose result is complete with fewer than 61 completed episodes (a
policy recorded as not scored, in the offline step) is flagged by the kit's progress and
verification scripts before its results are committed. A not-completed episode counts against the
claim at stake, in every test and in every row, title and fixed wording of sections 7.8 and 9:
whatever the episode would have given, it never makes a test easier to pass, its own candidate's or
another candidate's, never gets a candidate carried to confirmation that would not then have been
carried, and never licenses a row, a title or a wording that the completed episode could have ruled
out. It counts:
- in H1 and in confirmation, as a failure for a candidate and as a success for a comparator;
- in the family of H1's screen (7.3), as a discordant episode of every pair it belongs to. It can
  keep a candidate in the family, and so keep K up, but never take one out: one candidate's
  missing episode, or its comparator's, never lowers another candidate's bar;
- in m, which sets confirmation's level 0.05/m (7.5), so that m is at least what any outcome of
  the missing episodes would make it: m counts, up to three, every candidate that would clear the
  screen had each of its own missing episodes been a success and each of its comparators' a
  failure, against the screen's family with every episode that has a missing side counted as
  concordant;
- in the carried list (7.5), so that a candidate is carried only if it would be carried whatever
  the missing episodes gave: its rank in the carry order, with its own missing episodes as
  failures and its comparators' as successes, must be third or better against every other
  candidate that m counts, each with its own missing episodes as successes and its comparators'
  as failures. Fewer than three may then be carried. A candidate that is not carried but might
  have been had the missing episodes completed (in its best case it ranks third or better against
  the candidates that clear, each in its worst case) is listed: as "not carried: rank depends on
  missing episodes" if it clears, and as "not cleared: whether it clears depends on missing
  episodes" if it does not clear but m counts it. With nothing missing the carried list is the one
  the order of 7.5 gives, and nobody is listed;
- in the rows of section 9: row 2's first reading (training removed something the untrained model
  had) is given only if every trained candidate is decided whatever the missing episodes gave:
  none is listed above, and none that was carried and not confirmed would be confirmed with its
  own missing confirmation episodes as successes and its comparators' as failures, at 0.05 over
  the number carried. Otherwise row 2 keeps only its second reading (section 9). Row 5 needs every
  episode of the 37 completed; otherwise its statements are made of the N completed episodes, as
  H3's are;
- in the titles of section 9: the row-5 title needs every episode of the 37 completed, and the
  largest success count k that a title may give counts each policy's not-completed episodes as
  successes;
- in the wording of 7.8: "agree up to near-ties" needs all 366 pairs compared; otherwise it is said
  of the N compared pairs;
- in H2 (7.4), as a failure for rule B's picks and as a success for rule A's picks; in the
  description of rules C, D and G against rule A (section 8, O1), the same way;
- H3 is judged on completed episodes only, with the missing ones counted beside it (7.6).

In the descriptive outcomes of section 8 it counts as a failure, and its secondary outcomes are
reported as missing.

**Comparators.** The rule's cost falls on the six selected checkpoints. One comparator episode not
completed, which its candidates failed, raises the H1 bar of every candidate of its run: from 5 to
7 successes when the screen of 7.3 tests at 0.05, from 10 to 13 when it tests at 0.05/31. Update 0
is tested against all six (7.2). So the kit submits the six selected checkpoints before any
candidate, in every step (offline, main and confirmation), and within one job start retries an
error of one of them three times, against once for a candidate. Neither changes the rule above: any
episode, a comparator's included, is recorded as not completed after three errors over at least two
job starts. And the analysis lists, beside H1 and beside each confirmation test, every comparator
episode it counted as a success because it was not completed.

## 6. The smoke

The smoke runs on the study GPU before the freeze commit, from the code the freeze will carry. It
drives no policy on the main or the confirmation suite; what phase 4's controls show of those
suites is stated with phase 4.

- **Episodes.** Eight `val_seen` episodes in neither the main nor the confirmation suite: the four
  trajectories with the smallest ids among the 137 left (93, 139, 149 and 180), and for each its two
  lowest episode ids, in that order: episodes 4, 5, 10, 11, 13, 14, 16 and 17, in scenes
  `E9uDoFAP3SH`, `759xd9YjKW5`, `sT4fr6TAbpF` and `JeFG25nYj2p`, two consecutive episodes per scene.
- **Checkpoint.** wt25-u192, whose Spark rollout holds the most fragile decisions (section 2).
- **Phase 0.** The offline scorer scores wt25-u192 on the three held-out surfaces (1,152 decisions),
  and the scorer check of 7.4 compares every decision with the training records, which exist for
  all three surfaces at this selected checkpoint. Its training values are already published, so
  this previews nothing; it shows, before anything is frozen, whether the scorer on the study GPU
  reproduces the records closely enough for the offline step to be used. Because 1,152 decisions
  say little about the 14,848 of the main check, the smoke also checks two tighter bars here,
  which the main check does not use: no token NLL differs from its record by more than 0.5 nats,
  and no action changes at a decision whose recorded top-two margin is 0.75 nats or more. They do
  not decide whether the smoke passes; they decide whether the offline criteria are used. If a
  tighter bar fails, H2, rules B, C and G and O3 are reported as not run, whatever the offline
  step later gives, and O2 falls back to the 8 cells of the training records, as when the check of
  7.4 fails (7.4); the offline step still runs, and its check is reported. The committed smoke
  result records which bars failed, and the analysis script reads them from there.
- **Phase 1.** One worker alone drives the eight episodes, stops after the third completed episode
  (episode 10), and resumes in a new invocation that starts with episode 11, in the same scene. It
  records the model's load time, seconds per decision, peak GPU memory, peak host memory and GPU use
  every 2 s, and checks that all 288 trainable tensors are FP32.
- **Workers per GPU.** P is the largest whole number up to the cap for which P times the larger
  of two peaks, phase 1's worker's and phase 0's scorer's GPU memory (nvidia-smi's figure for each
  process; where nvidia-smi gives none, torch's peak reserved memory, which is a floor and is
  recorded as one), is at most 90% of the GPU's memory; the memory bound is the same number without
  the cap. The cap is 4, or fewer where 90% of the GPU's memory holds fewer copies of the model's
  bf16 weights (16,722 MiB): 2 on the L40S. Every worker and every scorer holds those weights on
  the GPU, so on the L40S P is at most 2 under any cap, and the cap sets only the job's size. Every
  one-GPU job, the smoke and every job of the offline, main and confirmation steps alike, asks for
  4 cores and 32 GB of host memory per worker the cap allows, and at least 16 cores and 64 GB: 16
  cores and 64 GB on the L40S. Each worker runs 4 threads. One GPU's share of an L40S node is 32
  cores and 128,250 MB (`sinfo` on 2026-10-07: 128 cores and 513,000 MB for four GPUs); the 128 GB
  of the eighth and ninth drafts is 131,072 MB in Slurm's units, over that share, and on
  2026-10-09 the author accepted the proposal of 64 GB. The smoke's peak host memory (Slurm's
  MaxRSS, read after the job) is reported with its summary. The smoke result records what the smoke had, and every later job asks
  for exactly that, so that the smoke measures the job the sweep runs. The offline step uses the
  same P.
- **Phase 2.** P workers at once on one GPU, each driving the same eight episodes without a stop.
- **Phase 3.** P scorers at once on one GPU, each scoring wt25-u192, as the offline step runs them.
- **Phase 4: the scene and simulator check.** No model is loaded, so it runs first, before phase
  0, and keeps its name. It starts no run after 0.75 h and is stopped 15 minutes later, so it
  cannot take the time of phases 0 to 3; if it reaches that cap, phases 0 to 3 still run and the
  smoke fails, naming phase 4's cap. Two
  non-learning controls,
  expert replay (the released ground-truth actions) and always-forward, drive every episode of the
  main suite (61) and of the confirmation suite (61) on the study GPU. They run through the code of
  the Spark's non-learning control suite `policy_ordering_v1` (CTRL-VALSEEN-v1, 854 runs): its
  episode loop, environment, policies and metrics, each file hashing as that suite's manifest
  records it. The environment has the sweep's configuration. The main suite runs in its own order,
  which is by scene, as the sweep drives it; the confirmation suite runs by scene. The Spark's
  values are that suite's reports of the 61 main episodes, whose per-control means must equal its
  pinned `suite.json` before anything runs. Neither control is a policy this file tests, and on
  the main suite the Spark's results for both are published. On the confirmation suite the check
  previews part of the comparators' secondary outcomes: on the Spark the three unweighted selected
  checkpoints took always-forward's actions in 61, 61 and 60 of the 61 main episodes (section 2),
  so always-forward's final distances and oracle successes there approximate theirs. No test, bar
  or decision of this file uses a secondary outcome (7.1). The check previews no candidate's
  outcome.

**Pass, all required:** the image's own test reports 0 errors on the primary base; every invocation
saw one GPU name, of the named type; the 14 snapshot files, the release files, the 1,152 cache
objects and the 37 checkpoint files hash as expected; all 288 trainable tensors are FP32 at every
load; the scorer scores wt25-u192 and passes the scorer check of 7.4; all eight episodes complete
in phases 1 and 2, in every worker; all P scorers of phase 3 score; no scorer or runner process of
phases 0 to 3 went more than 15 minutes, half the watchdog's 30 (4.3), without updating its
progress record, and every such gap was measured; phase 4 passes: on the main suite,
expert replay runs on all 61 episodes, and in each it succeeds and stops with the Spark's step count
and ends within 0.25 m (one forward step) of the Spark's final distance to the goal, and
always-forward runs on all 61 to the 500-step cap; on the confirmation suite, every one of the 61
episodes loads, resets and runs under both controls; and phase 2's mean GPU use, sampled every 2 s
while all P workers drive, is at least the node's cancellation line plus 10 points. The kit reads
the line from the node's name, as NYU's "Submitting jobs" page gives it: 50% on `ga`, `gl` and `gr`
nodes, so 60% is required there, and 60% on `gh` nodes and on any node whose name the kit does not
know, so 70% is required there. The L40S nodes are expected to be `gl` nodes, which the runbook
checks with `sinfo` before the smoke. The cluster cancels jobs whose GPU use stays below the line,
and P cannot change after the freeze commit, so this is checked now.

**Reported, not required:** how many of the eight episodes have identical action sequences across
the P workers, and between the resumed worker of phase 1 and the workers of phase 2; whether the P
scorers of phase 3 give phase 0's scores; the measured times, memory and GPU use (phase 3's use
with a warning below the same line: offline jobs take minutes per policy, and a cancellation is an
interruption, section 5), the scorer check's reported values and the projected hours; for both
controls on the main suite, the largest differences from the Spark's final distance and from its
position at any step, the number of episodes whose trace is bit-identical to the Spark's and the
number with a step more than 1e-4 m apart; and how many confirmation replays succeed (on the Spark,
all 778 `val_seen` replays did). If the workers or the resume change a rollout, the main run still
goes ahead, because every test takes each rollout once, as it came, and no resume is triggered by an
outcome; but the paper then says that a rollout on the study GPU is one draw, not a fixed function
of the checkpoint.

**If the smoke fails,** nothing is frozen until the cause is found and a smoke passes; the summary
of every smoke run, passed or failed, is reported with the results. Fixed now:
- *GPU use below the line.* Before the freeze commit, the author either raises the cap on P above 4,
  up to the memory bound, and runs the smoke again, or names another GPU type and runs the smoke
  there. If the memory bound is already P, only the second remains; on the L40S it always is.
- *Host memory.* If Slurm ends the smoke for host memory (state `OUT_OF_MEMORY`), the host memory
  per worker is raised, at most to one GPU's share of the node, and the smoke runs again from the
  new commit.
- *Phase 0.* If the bars of 7.4 fail, the port is taken to be broken (a wrong image, cache object or
  preprocessing): it is fixed and the smoke runs again from the new commit. If no fault is found,
  the author names another GPU type and runs the smoke there. A failure of the two tighter bars
  alone is not a failed smoke (phase 0, above).
- *Phase 4.* A scene or an episode that does not load, or a main-suite control run that fails the
  bars above (a replay that does not succeed and stop with the Spark's step count, or ends more than
  0.25 m from the Spark's final distance; an always-forward run short of the cap), is first taken
  to be a fault of the data or of the port. It is looked for, and a fault found is fixed and the
  smoke runs again, from the new commit if the code changed. If no data or port fault is
  found, the smoke result records every miss with its distance, the author decides before the
  freeze whether the sweep runs, and the paper reports the misses (the author's rule, confirmed
  on 2026-10-09). Another GPU type is no remedy:
  the image is built for amd64 (4.3), so every node it runs on is x86_64 whatever its GPU, and the
  Spark is aarch64. The risk looks small: on the Spark the 61 main replays had no collision, so no
  step was resolved against an obstacle, and each ended 0.260 to 0.497 m from the goal, well inside
  the 3 m of success.
- *Phase 4 reaches its cap with no fault found.* Before the freeze the author may raise the cap,
  which changes this file and `lib.sh`, and the smoke runs again from the new commit.
- *Anything else* (an episode that does not complete, a dtype, a name, a hash): the fault is found
  and fixed, and the smoke runs again from the new commit.

## 7. Outcomes, hypotheses and tests

Every closed-loop outcome below comes from the main run or the confirmation run on the study GPU.
Every offline criterion comes from the offline step on the same GPU type, or, where 7.4 says so,
from what training recorded. The Spark recordings enter only the cross-check of 7.8.

### 7.1 Outcomes

- **Primary: success** in each of the 61 episodes, yes or no, for each checkpoint. SR is successes
  divided by 61, with an exact two-sided 95% Clopper-Pearson interval.
- **Secondary:** STOP events, split into those inside 3 m (the successes) and outside, with the
  distance and step of each press and whether it was an exact tie; final navigation error, mean and
  median, in metres; oracle success, as a count of 61. A STOP press is an exact tie if, at the
  decision that chose it, another action's token NLL equals STOP's: the tie went to STOP (4.1), and
  STOP was not strictly preferred, the reading CR-052 uses for STOP recall.
- **Reported without a test:** SPL, nDTW, SDTW, step-cap rate, collisions, action histogram,
  constant-action episodes, match to the always-forward control, and closed-loop exact ties. No
  secondary outcome decides whether a checkpoint is better, and nDTW does not break ties. Every
  outcome of 7.1 that the analysis script does not compute (among them the median navigation
  error, constant-action episodes, the match to the always-forward control and closed-loop exact
  ties at decisions other than a final STOP) comes from the committed episode reports, by a
  descriptive script that decides nothing.

### 7.2 Comparators

The comparator of a trained candidate is its own run's selected checkpoint. Update 0 stands for the
update-0 file of all six runs (section 3), so its comparators are all six selected checkpoints, and
it is tested against each: its p is the largest of its six p-values, and it clears a bar only if it
clears it against every one. This is an intersection-union test: the claim needs all six tests to
reject, so each is run at the full level, with no correction for the six (Berger, Technometrics
24:295-300, 1982). The eighth draft compared update 0 with uw25-u160 alone, so row 2 could have
said that training removed something the untrained model had while another run's selected
checkpoint succeeded more often than update 0. Comparing update 0 with the union of the six (an
episode counts if any of them succeeds, H2's E_A) is not used: it asks update 0 to beat the six
together, which is more than row 2 claims, and its p is never below the intersection-union p,
because the union can only remove episodes that update 0 wins alone and add episodes it loses.

Every comparator is the selected checkpoint as driven in the main run on the study GPU, never its
Spark recording.

### 7.3 H1: a better checkpoint existed (primary)

For each candidate j among the 31, the null hypothesis H0_j is that j succeeds on no more episodes
than its comparator, the two paired by episode; for update 0, that it succeeds on no more episodes
than at least one of its six comparators (7.2).

- **Test: the one-sided exact McNemar test,** a sign test on the episodes in which exactly one of
  the pair succeeds. With b_j the number of episodes only j succeeds on and c_j the number only its
  comparator succeeds on, p_j = P(Binomial(b_j + c_j, 1/2) >= b_j). If the comparator succeeds
  nowhere, as on the Spark, then c_j = 0 and p_j = 0.5^k_j, where k_j is j's successes. For update
  0, p_j is the largest of its six p-values. Fisher's exact test is not used: the two sets of 61 are
  the same episodes.
- **The screen: Tarone's discrete Bonferroni, in its alpha-consistent form,** over the 31
  candidates at a family-wise level of 0.05. With b_j + c_j discordant episodes, the smallest p the
  test can give is p*_j = 0.5^(b_j + c_j) (for update 0, the largest of its six); a candidate that
  succeeds nowhere against a comparator that succeeds nowhere has p*_j = 1. For p*_j only, every
  episode on which the candidate or its comparator was not completed counts as discordant (section
  5): that can only enlarge the family, so it lowers no bar. For a level g, let
  M(g, k) be the number of candidates with p*_j <= g/k, and K(g) the smallest k from 1 to 31 with
  M(g, k) <= k. Tarone's procedure (Biometrics 46:515-522, 1990) rejects H0_j if
  p_j <= 0.05/K(0.05). This file uses its alpha-consistent form (Hommel and Krummenauer,
  Biometrics 54:673-681, 1998; Procedure 2.2 of Zhu and Guo, arXiv 1711.08147): H0_j is rejected,
  and j clears the H1 bar, if some g with 0 < g <= 0.05 has p_j <= g/K(g). K can change only at the
  points g = k p*_i, so the analysis script, in exact fractions, clears j if p_j <= 0.05/K(0.05),
  or if p_j < g/K' at one of those points g <= 0.05, where K' is K just below g (M counted with
  p*_i < g/k).
- **Example,** with every comparator at 0: one candidate succeeds 7 times, one 6 times, one 5 times,
  the rest never. Three have p* <= 0.05, so K(0.05) is not 1; at k = 2 only the first two have
  p* <= 0.025, so K(0.05) = 2 and the threshold is 0.025. The first two clear (p = 1/128 and 1/64);
  the third (p = 1/32) does not, at 0.05 or at any lower level. Alone, the third would clear: K = 1.
- **Why this screen.** The eighth draft corrected twice: Bonferroni over 31 at the screen, then a
  confirmation at 0.05/m on fresh episodes, which by itself guards the claim. Most candidates are
  expected to succeed nowhere, and a candidate that cannot reach significance whatever the outcome
  leaves Tarone's family, so the screen pays only for candidates that could clear. K(0.05) is never
  above 31, so the threshold lies between 0.05/31 and 0.05, and the screen never needs more
  successes than Bonferroni over 31 did. Tarone's procedure keeps the family-wise error at 0.05
  when the pairs are independent: given the discordant counts, each p_j is then a valid p-value
  whatever the other pairs gave, and Bonferroni over the at most K candidates of the family does
  the rest. Here the pairs are not independent. Five or six candidates share each comparator, and
  update 0 is tested against all six, so one pair's discordant count says something about another
  pair's split, and the guarantee rests on simulation (7.7), not on proof. Tarone's procedure is
  also not alpha-consistent: it can clear a candidate at a level below 0.05 and fail it at 0.05
  (the CPU tests hold a case: b = 11, c = 3 against four candidates with p* = 1/32; it clears at
  0.03 and not at 0.05). The alpha-consistent form clears everything Tarone's procedure clears at
  0.05, and anything it clears at a lower level. Its rejections are those of Tarone's procedure at
  one level g, not above 0.05, the one with the largest g/K(g), or at levels just below it, so it
  has Tarone's guarantee under the same condition. Row 3 of section 9 states that a candidate
  clears on the main suite as a fact in its own right; a family-wise screen is what makes that
  statement defensible, with the limit just stated, and a screen without correction would not.
- **The bars,** the smallest b_j that clears the threshold 0.05/K(0.05), with every comparator at
  0: 5 when K = 1 (no other candidate could reach 0.05), 6 for K = 2 or 3, 7 for 4 to 6, 8 for 7 to
  12, 9 for 13 to 25, and 10 for 26 to 31, the eighth draft's bar. Appendix A gives the bars for
  c_j = 1 to 4; with c_j = 1 they run from 7 (K = 1) to 13 (K = 30 or 31). The alpha-consistent form
  clears a candidate above 0.05/K(0.05) only when a lower level has fewer candidates in its family,
  as in the case of the CPU tests above.
- **Decision:** H1 holds on this suite if any candidate clears the screen. It becomes a claim only
  after confirmation (7.5).
- **The tie-STOP sensitivity.** Exact ties go to STOP (4.1), and the study GPU can break a near-tie
  differently from the GB10 (section 2), so a rounding difference can make a STOP inside the goal
  radius that the policy did not prefer. The analysis runs H1 a second time with every success of a
  candidate whose final STOP was an exact tie (7.1) counted as a failure, each candidate's test
  recomputed against the main test's K, and recomputes confirmation (7.5) the same way, at the main
  test's m. The family is not recounted: had it been, a candidate whose successes all ended in
  exact ties would leave it, and K, and with it another candidate's bar, could fall. Comparators'
  successes do not change, so that, like a missing episode, a tie never makes a test easier to
  pass. Every success that ended in an exact-tie
  STOP is listed. The row of section 9 is decided by the main test; a row-1 or row-2 claim states
  whether its candidate also clears the screen and confirmation in this recomputation ("survives
  the tie-STOP sensitivity") or not.
- **Why one-sided:** the hypothesis is directional and fixed before any rollout; a two-sided test
  would double every p to cover a direction no claim uses.

For reference, with the comparator at 0: one-sided at 0.05, 5 (p = 0.031); two-sided at 0.05, 6;
one-sided with Bonferroni over 31, 10; two-sided with Bonferroni over 31, 11.

### 7.4 H2: the STOP-aware selector (the one pre-named comparison)

**Rule B.** Within each run, pick the checkpoint with the lowest K3, the class-balanced CE on the
256 balanced decisions (defined in section 8), with ties going to the earlier update. All seven
checkpoints of each run are eligible, update 0 included, as they were for rule A. Rule A is the
preregistered rule, and its picks are the six selected checkpoints as recorded (section 8).

**Where rule B was named.** `D4_final.md`, line 40: "*Rule B better?* Its six picks need 5 or more
successes in 366 episode runs against A's 0". That file was written before any unselected checkpoint
was driven or scored (scratchpad `rejected_paper/designs/D4_final.md`, sha256
`89c4090d3868d1faeaf982111cf41871b0ffcec6f7d5bd2f0c788f569e8bd84d` on 2026-09-30). The file cannot
show that rule B was named before the six selected checkpoints' balanced scores were known. That
does not matter for H2: the other 31 policies' balanced scores do not exist until the offline step,
so nobody could have tuned rule B to them.

**Picks.** The analysis script computes rule B's six picks, and those of rules C and G, from the
offline step's records, by the rules of this file. Nobody chooses or writes a pick by hand, and the
picks are fixed by offline scores before any outcome of the main run is read.

- **Statistic.** E_B is the number of the 61 episodes on which at least one of rule B's six picks
  succeeds. E_A is the same count for rule A's picks, the six selected checkpoints as driven in the
  main run on the study GPU.
- **Test.** The one-sided exact McNemar test at the episode level. Let b be the number of episodes
  on which only rule B's picks succeed, and c the number on which only rule A's do. Then p =
  P(Binomial(b + c, 1/2) >= b). If rule A's picks fail every episode, as on the Spark, c = 0,
  b = E_B and p = 0.5^E_B.
- **Bar.** p <= 0.05, one comparison, no correction: E_B >= 5 when c = 0 (p = 0.031; four give
  0.0625 and fail), and b of 7, 9 or 10 when c is 1, 2 or 3.
- **Missing episodes.** An episode that a rule-B pick did not complete counts as a failure of that
  pick, and one that a rule-A pick did not complete as a success of that pick (section 5).
- In a run where rule B picks the same checkpoint as rule A, that run contributes the same rollout
  to both sides. In a run where rule B picks update 0, it contributes uw25-u0's rollout. The number
  k of runs in which the two rules pick the same checkpoint is reported with E_B. If k = 6, then
  b = c = 0 whatever the rollouts, and H2 is reported as "not informative: same picks in 6 of 6
  runs", not as failed.
- Rule D, the last checkpoint, is reported beside rule B with its own E_D (section 8, O1).
- The scene-level version of the same test is reported beside it and does not decide (7.7).

**Why H2 needs no confirmation run.** Offline scores fix its picks before any rollout, so selection
on this suite's outcomes cannot inflate them.

**The scorer check, which decides whether the offline step is used.** The offline step re-scores
every decision that training recorded: the selection surface at all 37 policies (37 x 256 = 9,472
decisions; the update-0 scores are equal in all six runs, so uw25-u0's are compared) and the
natural and balanced surfaces at the six selected checkpoints (6 x 896 = 5,376): 14,848 in all.
The study GPU is not the GB10 that made the records, so bit-identity is not expected. A BF16 logit
step at these magnitudes is 0.125 nats, and a change of GPU can move a logit by a step or a few,
which flips decisions that were near-ties (2,504 of the 14,848 were won by 0.25 nats or less) but
not decisions won by a whole nat (8,571 of the 14,848). The check passes if both hold:
1. at every compared decision whose recorded top-two margin is 1.0 nats or more, the chosen action
   is unchanged;
2. for every compared policy and surface, the mean restricted-action CE differs from the recorded
   one by at most 0.05 nats.

Reported in every case: how many decisions are bit-identical; the largest token-NLL difference; the
largest mean-CE difference; and the changed actions by recorded margin (0, 0.125, 0.25, 0.375 to
0.875, and 1.0 or more). The bars 1.0 and 0.05 are chosen without data on this GPU change; they are
set to catch a broken port (a wrong image, a wrong cache object, a different preprocessing), not
the hardware's rounding. The smoke's phase 0 (section 6) runs the same check on wt25-u192 before
the freeze commit, when a broken port can still be fixed, with two tighter bars of its own; if those
fail, H2 and what rests on the offline step are not run (section 6).

**If the check passes, every policy is scored and the smoke's tighter bars passed,** K1 to K6 of all
37 policies come from the offline step, so every criterion comes from one code path on one GPU type,
and H2, rules B, C and G, O2's 20 cells and O3 are computed (section 8). Rule A keeps its recorded
picks.

**If the check fails, a policy could not be scored, or the smoke's tighter bars failed** (a policy:
three failed attempts, section 5; the bars: section 6), no fix is possible after the freeze commit.
H2, rules B, C and G and O3 are reported as not run, the policies not scored are named, and O2
falls back to the criteria training recorded for all 37: K1 and K5 on the selection surface, 8
cells (section 8). H1 does not depend on the offline step and goes ahead.

### 7.5 Confirmation

**Which candidates.** The candidates that clear the H1 bar are carried, up to three, in this order:
smallest p_j, then most successes, then the earlier update, then the run order of section 3. There
is no override. Candidates past the third are reported as clearing the bar on the main suite and not
confirmed (section 9, row 3). If an episode of the main suite was not completed, a candidate is
carried only if it is among the first three whatever the missing episodes gave (section 5); fewer
than three may then be carried. A candidate that clears the bar and is not carried, but might have
been had the missing episodes completed, is reported as "not carried: rank depends on missing
episodes" and, like a candidate past the third, as clearing the bar on the main suite and not
confirmed. The analysis script writes the carried list.

**Which comparators.** Each carried candidate's comparators of 7.2 (all six selected checkpoints if
update 0 is carried), each driven once even if it serves two candidates. All of them run on the
study GPU of the main run.

**Test.** The one-sided exact McNemar test on the confirmation suite's 61 episodes at alpha 0.05/m.
With every main-suite episode completed, m is the number of carried candidates. If some were not
completed, m counts, up to three, every candidate that would clear the screen of 7.3 had each of
its own missing episodes been a success and each of its comparators' a failure, against the
screen's family with every episode that has a missing side counted as concordant. So m is never
below the number carried, and a missing episode never lowers it (section 5). The analysis script
writes m into the carried list. If the comparator succeeds nowhere there, the bar is b >= 5 for
m = 1 and b >= 6 for m = 2 or 3; if it succeeds on one episode on which the candidate fails, the
bar is 7, 8 or 9 for m = 1, 2 or 3. Update 0 is confirmed only if it clears against each of its
six comparators at 0.05/m. The same test with scenes as the unit is reported beside it and does
not decide, and the test is recomputed for the tie-STOP sensitivity of 7.3.

**The confirmation suite,** built before the freeze commit by the existing builder: from the
`val_seen` release (778 episodes, 259 trajectories, 53 scenes), remove the main suite's 61
trajectories (198 remain), draw `random.Random(20260930).sample(sorted(ids), 61)`, and take for each
drawn trajectory its lowest integer episode id, in draw order. The dry run of 2026-09-30, reproduced
on 2026-10-04, gives 61 trajectories in 34 scenes, first ten ids 4762, 4087, 6158, 7130, 2328,
6472, 3190, 6247, 5098 and 6943; the build must reproduce it. Its median release geodesic is 7.9 m
against 12.5 m for the main suite, so it is easier and more clustered: only the paired comparison
inside it counts, never a comparison of rates across the two suites.

### 7.6 H3: a prediction from the offline STOP counts

All 18 trained unweighted checkpoints predict STOP 0 times on the selection surface. The prediction
is that the 15 unselected ones press STOP in 0 of 61 episodes each. One press anywhere falsifies it;
there is no p-value. It is judged on completed episodes only, and if any are missing the result is
stated as "no STOP press in the N completed episodes", never as "0 of 61". The three selected ones
pressed STOP in no episode on the Spark; their counts on the study GPU are reported beside H3.

### 7.7 Dependence, power and error rates

- **Dependence.** The 61 episodes come from 53 scenes (47 with one episode, 4 with two, 2 with
  three), so McNemar's pairs are not fully independent. The same sign test is run with scenes as the
  unit (a scene counts if any of its episodes succeeds), and its p is held to the threshold the
  screen of 7.3 set for the episode-level test. The decision uses the episode-level test; if the
  scene-level test does not also clear, the paper says so next to the result. The confirmation
  suite is more clustered: 61 episodes from 34 scenes (12 with one episode, 19 with two, one with
  three, two with four). The same scene-level test is computed there, at the confirmation's own
  threshold, and reported beside every confirmation result in the same way.
- **Power,** when every comparator succeeds nowhere. A candidate whose true success rate is 5%, 8%,
  10% or 15% reaches 5 of 61, the bar when it is the only candidate that could reach 0.05, with
  probability 0.19, 0.54, 0.74 or 0.96; the eighth draft's bar of 10, which applies only when K is
  26 or more, is reached at 10%, 15%, 20% or 25% with probability 0.08, 0.43, 0.80 or 0.96. In the
  setting of the outside reading (one candidate at rate r, the other 30 at 2%, every comparator at
  0, at most three carried, confirmation on 61 fresh episodes at 0.05/m with the same rate), the
  chance that the candidate is confirmed, computed exactly by enumeration, is:

  | true rate r | 5% | 8% | 10% | 15% | 20% | 25% |
  |---|---|---|---|---|---|---|
  | this screen | 0.031 | 0.274 | 0.523 | 0.913 | 0.992 | 0.9995 |
  | Bonferroni over 31 (eighth draft) | 0.0002 | 0.012 | 0.059 | 0.417 | 0.801 | 0.961 |

  So the study has about even odds of confirming a checkpoint that succeeds in 10% of episodes, and
  will usually miss one whose rate is 8% or less. A null H1 does not mean that no checkpoint beats
  0%. If a comparator succeeds, its candidates need more successes, and these chances fall.
- **Error rates.** Tarone's guarantee of a family-wise error of at most 0.05 holds for independent
  pairs (7.3). Here candidates of one run share a comparator and update 0 is tested against all
  six, so the pairs are dependent, and that the screen keeps the family-wise error of H1 on the
  main suite at 0.05 rests on simulation, not on proof. In the simulation run for the ninth draft,
  with every null true, the screen rejected at least one H0_j in 1.9% to 2.3% of 4,000 replicates
  in each of five settings: all 37 policies with one success probability on every episode (2%, 10%
  or 30%), or with a probability drawn for each episode and shared by all 37 (mean 10% or 30%). In
  a second reader's simulation of seven adversarial null settings, with update 0 tested against all
  six, it rejected in 0.7% to 2.1%. An H1 claim also needs confirmation at 0.05/m on fresh
  episodes, so, treating episodes as independent, its own false-positive rate is below 0.05.
  Episodes in one scene can share their outcome, and then this bound need not hold: in a case
  built with two main and two confirmation scenes, every null true and each scene's outcome shared
  by all its episodes, the procedure claims H1 with probability 0.0625 (CR-054). The paper states
  the independence assumption beside any H1 claim, with the scene-level tests above. H1 and H2
  are each tested at 0.05, so, under the same assumption, the chance of at least one false
  positive across the two is up to about 0.10. The cells of O2 (section 8) are tested at 0.05/20 each (0.05/8 in the fallback of
  7.4).

### 7.8 Cross-check: the six selected checkpoints on the Spark and on the study GPU

For each of the six selected checkpoints and each of the 61 episodes, the analysis compares the main
run's rollout with the Spark recording (`native_c0_closed_loop_v1`, matched by
`legacy_checkpoint_id` and episode id) and reports: whether the action sequences are identical, and
if not, the first step where they differ with the top-two margin each machine recorded there;
whether success, oracle success and the stopped flag agree, and the difference in final distance;
and, over the decisions before the first difference, the largest difference in any token NLL. The
totals are the number of the 366 episode pairs with identical actions and with identical flags.

**Wording, fixed now.** If every episode whose actions differ first differs at a decision whose
Spark-recorded top-two margin was 0.25 nats or less, the paper may say that the two machines agree
up to near-ties. Otherwise it names the episodes that first differ at a larger margin and does not
use that phrase. The phrase also needs all 366 pairs compared: if a pair was not compared because
an episode was not completed, the paper says instead that in the N compared pairs the two machines
agree up to near-ties, and names the pairs not compared.

**What it decides: nothing.** The tests use only rollouts on the study GPU, and a disagreement stops
nothing. The check cannot say why two rollouts differ: the GPU, its kernels, the renderer and the
new code all differ at once. If a selected checkpoint's success differs between the machines in any
episode, the paper says so beside the published numbers; the published 0 of 61 stays the Spark's
result, and the study GPU's count is reported next to it with its interval.

## 8. Offline criteria against closed-loop outcomes, on all 37 policies

**Criteria.** The offline step scores every policy, on the study GPU type, as the first job after
the freeze commit, on the three held-out surfaces all six runs share: 256 selection decisions (173
FORWARD, 3 STOP, 42 LEFT and 38 RIGHT), 640 natural and 256 balanced (64 per action). It loads each
checkpoint with the closed-loop loader, checks that all 288 trainable tensors are FP32, scores each
surface with the training function `_evaluate` itself, and computes the existence gate with the
gate's own function. If the scorer check of 7.4 passes and every policy is scored, these values are
used for all 37:
- K1: teacher-forced CE on the 256 selection decisions, the preregistered selection rule's own
  criterion. Training recorded it for all 42 files;
- K2: CE on the 640 natural decisions;
- K3: class-balanced CE on the 256 balanced decisions: the observed mean restricted-action CE,
  `existence_gate.balanced_diagnostic.observed_mean_restricted_action_ce`, which with 64 decisions
  per action is the average of the four per-action CEs. It is not the bootstrap upper bound;
- K4: balanced STOP recall, out of 64, counted two ways: with ties going to STOP, as the scorer and
  the gate count it, and strictly, with ties excluded (the CR-052 convention);
- K5: STOP predictions on the 256 selection decisions, ties going to STOP, as in the section 3
  table;
- K6: the existence gate. It passes only if all three of its conditions hold: natural upper-95% CE
  below 0.8747; balanced upper-95% CE below 1.3363; and balanced STOP recall of at least 16 of 64,
  the gate's own count (ties to STOP). Each condition is also reported on its own.

If the offline step cannot be used (7.4), only what training recorded for all seven checkpoints
exists: K1 and K5, from `calibrations[k].summary` in each run's `result.json` (pinned by
`native_c0_closed_loop_v1`). Training recorded K2, K3, K4 and K6 only at each run's selected
checkpoint.

**Outcomes,** from the main run: Y1 successes, Y2 oracle successes, Y3 episodes ending in STOP, and
Y4 mean final navigation error. Y1 counts a not-completed episode as a failure (section 5); Y2, Y3
and Y4 count completed episodes only.

**Eligibility.** For every rule, all seven checkpoints of each run are eligible, update 0 included,
as they were for rule A (`selection.candidate_count = 7` in all six runs). Update 0's criteria are
shared by the six runs, and when a rule picks update 0, uw25-u0's rollout counts for that run.

**Units for O2.** The 36 trained checkpoints, six per run: 30 candidates and 6 selected, all driven
in the main run. Update 0 gets its own row and stays out of the correlations, because it belongs to
every run.

**O1: selection rules.** Each rule is applied within each run. For each, the six picks and their Y1
to Y4 are reported, with E, the number of episodes on which at least one pick succeeds:
- A: the preregistered rule. Its picks are the six recorded selected checkpoints (updates 160, 192,
  160, 128, 160 and 192 in the run order of section 3), whatever the re-scored K1 gives. The
  smallest recorded margin between rule A's pick and the runner-up is 0.0029 (wt26: u128 at 0.9389
  against u192 at 0.9418), so a re-score could otherwise move a pick;
- B: lowest K3, ties to the earlier update. H2 tests this rule (7.4);
- C: highest strict K4, with ties going to the lower K3 and then the earlier update;
- D: the last checkpoint, update 192, which is what one-epoch VLM recipes keep. In wt25 and wt27 it
  is the selected checkpoint;
- G: lowest K1 among the checkpoints that pass K6, ties to the earlier update. If none passes in a
  run, the rule deploys nothing there, and the number of runs where it abstains is reported.

Only B is tested. For C, D and G, E and its p against rule A by the formula of 7.4 are given as
description only, with the missing-episode rule of section 5, and every such p is labelled
"unadjusted, descriptive" where it is printed. The analysis script computes every rule's picks
from the offline records; nobody writes a pick by hand. Rules B, C and G are not computed if the
offline step cannot be used.

**O2: association.** K1 to K5 against Y1 to Y4: 20 cells (with the fallback of 7.4, K1 and K5
only: 8 cells).
- K4 is the strict count; the tie-to-STOP count is reported beside it, without a test. K5 is the
  tie-to-STOP count;
- the statistic T is the mean, over the runs kept in the cell, of the within-run Spearman
  correlation across the six trained checkpoints (Pearson correlation of average ranks);
- the null: within each kept run, each of the 720 orderings of its six outcomes is equally likely,
  independently across runs. T's null distribution is the convolution of the runs' distributions
  (each run's correlation takes at most 720 values), computed exactly;
- p is the null probability that |T| >= |T_observed| - 1e-12, two-sided, exact;
- p_min, the smallest p the cell can give, is the null probability of the largest |T| that any
  ordering reaches. It depends only on the kept runs and their ties: one run with six distinct
  outcomes, 2/720 = 0.0028; three runs with one success each, 2/216 = 0.0093; four such runs,
  2/1296 = 0.0015; three runs with two successes each, 2/3375 = 0.0006. It is reported beside p.
  The eighth draft estimated p from 10,000 random draws: a one-run cell, whose p cannot go below
  0.0028, would have been reported below 0.0025 with probability 0.27;
- a run in which the criterion or the outcome is constant is dropped from the cell, and the number
  dropped is reported; a cell with every run dropped is not computed. A cell of Y4 is not computed
  if any of the 36 trained checkpoints completed no episode at all, since its mean distance does
  not exist. K5 is 0 at every trained unweighted checkpoint in the training records, so its cells
  will probably use only the weighted runs;
- a cell whose p_min is not below the bar, 0.05/20 = 0.0025 (0.05/8 = 0.00625 in the fallback),
  is reported as "untestable": no outcome could have been called an association there. A testable
  cell is called an association only if p is below the bar; otherwise the paper says "no
  association detectable in 36 checkpoints". All cells are reported. The pooled Spearman over all
  36 is given as description, without a p-value.

**O3: warnings.** Two-by-two counts of K6 (pass or fail) against Y1 >= 1, and against Y1 >= 10. The
same counts are given for the STOP-recall condition alone: K4's tie-to-STOP count, the gate's own,
at 16 of 64 or more, with the counts for the strict K4 beside them. These are counts only, with no
test. They answer zx7k's point that the gate and the per-class recall already gave offline warnings:
the tables show how often a warning was followed by failure in closed loop, and whether any
checkpoint passed the gate at all. If the offline step cannot be used, O3 is not computed; training
recorded the gate only at the six selected checkpoints, all of which failed it (balanced STOP recall
0, 12, 0, 1, 0 and 6 of 64), and each scored 0 of 61 on the Spark.

**Descriptions,** fixed now, with no test and no decision. They give zx7k the size of the selection
effect, which H1's yes or no does not. Y1 counts a not-completed episode as a failure (section 5).
- **Selection regret, per run:** the most successes (Y1) among the run's seven checkpoints, update
  0's rollout included, minus the selected checkpoint's, with the checkpoints that reach the most.
  It is biased upward, because it is the largest of seven counts on the same 61 episodes, and is
  labelled so. Only confirmation (7.5) gives an unbiased difference.
- **Episodes solved by any policy:** the number of the 61 episodes on which at least one of the 37
  policies succeeds, with their ids. Over the completed episodes, no rule that picks among these
  checkpoints could have reached a larger E; the number of episodes not completed is given beside
  it.
- **The largest success count** among the 37 (Y1), and k, the same count with each policy's
  not-completed episodes counted as successes, which a title of section 9 may give.
- **The family bound:** for each policy, the one-sided upper Clopper-Pearson bound on its success
  rate at level 0.05/37, from its successes among its completed episodes. The largest of the 37 is
  a simultaneous one-sided 95% upper bound on all 37 rates (Bonferroni, valid whatever the
  dependence). With 0 of 61 everywhere it is 1 - (0.05/37)^(1/61) = 10.3%; for one checkpoint
  alone, the one-sided 95% bound on 0 of 61 is 4.8%, and a checkpoint whose true rate is 10% scores
  0 of 61 with probability 0.9^61 = 0.0016.

## 9. What each outcome means for the title and the claim

The current title is "Teacher-Forced Action Loss Can Select Policies That Never Stop in
Vision–Language Navigation". The outcomes are checked in order, and the first that applies decides.
"Clears the H1 bar" means that the screen of 7.3 rejects H0_j: with every comparator at 0, 5 to 10
successes of 61, depending on how many candidates could reach significance; for update 0, against
each of the six selected checkpoints.

| # | condition | title | claim |
|---|---|---|---|
| 1 | A trained candidate (update 32-192) clears the H1 bar and passes confirmation | stands, with "never stop" limited to the goal (rule below) | The loss passed over a checkpoint that navigates. The paper names it, its run and update, and its successes on both suites, with intervals, against its comparator's counts on both suites. This becomes the paper's main result. If update 0 also passes, both are reported. It states whether the candidate survives the tie-STOP sensitivity of 7.3 and lists its successes that ended in an exact-tie STOP. The scope is unchanged: one backbone, one budget, buildings seen in training |
| 2 | Update 0 clears the H1 bar and passes confirmation, and no trained candidate does | changes | Two readings are reported, both true. Training at this budget removed something the untrained model had, which is jsSy's reading: update 0 succeeded on more episodes than each of the six selected checkpoints, on both suites (7.2). And the loss ranked the untrained model last of the seven in every run (CE 5.49, the worst of the 42), so no rule ranking by this loss would have kept it, which is a selection result. The claim states whether update 0 survives the tie-STOP sensitivity of 7.3. If a trained candidate is undecided by missing episodes (section 5: listed as not carried or not cleared because of them, or not confirmed only because of its own missing confirmation episodes), the first reading is not given, because that candidate, completed, could have been confirmed. The paper gives the second reading and update 0's counts against each selected checkpoint, which hold whatever the missing episodes gave, names that candidate, says that whether a trained checkpoint does better than the selected one is not decided, and takes its title by the rule for rows 3 and 4 (the analysis reports "2, selection reading only") |
| 3 | A candidate clears the H1 bar on the main suite but is not confirmed: it fails confirmation, or it was not carried (a fourth or later candidate, or one whose rank depends on missing episodes, section 5) | changes, by the rule for rows 3 and 4 below | The counts on both suites (where driven) are reported with intervals, and no checkpoint is called better |
| 4 | Some of the 37 policies succeed in some episode on the study GPU, but no candidate clears the H1 bar | changes, by the rule for rows 3 and 4 below | Every count is reported with its interval. The paper says "no checkpoint was shown to navigate better than the selected one", not "none navigates" |
| 5 | None of the 37 policies succeeds in any episode on the study GPU | changes | Every saved checkpoint of the six runs, 37 distinct policies (42 files; the six update-0 files count as one because every `lora_B` is zero), scores 0 of 61 on the study GPU, with an upper 95% bound of 0.059 each. Training at this budget produced no policy that stops inside the goal radius on this suite. The averaged loss fell from 5.49 to 0.73-0.75 in the unweighted runs and to 0.92-0.94 in the weighted runs, and it did not flag this. Any episodes recorded as not completed are counted and named beside this. The family bound of section 8 is stated: a checkpoint whose true rate is 10% scores 0 of 61 with probability 0.0016, and the simultaneous one-sided 95% upper bound on the 37 success rates is the one the analysis reports, 10.3% if every policy completed its 61 episodes (4.8% for one checkpoint alone). The gate and the STOP recall are reported from O3 as counts, with the recall as the gate counts it (ties to STOP): "the gate failed at k of 37 policies and the STOP recall fell below 16 of 64 at n; m passed and still scored 0 of 61" (if the offline step could not be used: the gate was computed only at the six selected checkpoints, where it failed every time). There was no better checkpoint to pick, so the selection rule is not shown to be the cause. If any episode of the 37 was not completed, every statement of this row is made of the N completed episodes ("none of the 37 policies succeeds in any of the N completed episodes"), and the title is not row 5's but follows the rule for rows 3 and 4 (the analysis reports "5, completed episodes only") |

**What any title may and may not say** (the wording is the author's **[!]**), in every row:
- it must not say or imply that the selection rule chose badly, except as H2 licenses (below);
- it must not say "never stop" without "at the goal" or "inside the goal radius", because the
  weighted selected policies pressed STOP 7 times on the Spark, each far from the goal. Under row 1
  the current title stands only with this change;
- it may say that the loss improved while no checkpoint stopped at the goal only in row 5 with
  every episode of the 37 completed, when all 37 policies score 0 of 61 (section 8). One
  candidate for row 5: "Teacher-Forced Action Loss Improves While No Checkpoint Stops at the Goal in
  Vision–Language Navigation";
- in rows 3 and 4, in row 2 reduced to its second reading, and in row 5 with an episode not
  completed, some policy succeeded in some episode or might have, so the title does not say "no",
  "none" or "never" of stopping at the goal. It may give k, the largest success count among the 37
  on the main suite with each policy's not-completed episodes counted as successes (section 8).
  One candidate for rows 3 and 4: "Teacher-Forced Action Loss Improves While Its Checkpoints Stop
  at the Goal in at Most k of 61 Episodes in Vision–Language Navigation".

**On top of any row:**
- H2: if it passes, the conclusion may say that on this suite, picking by class-balanced CE
  produced successes where picking by averaged CE produced fewer. It stays one suite and one budget,
  with no general claim about selectors. If E_D >= E_B, or rule D's six picks are the same as rule
  B's, the paper also says that picking the last checkpoint did as well, and that class balance is
  not shown to be the reason. If H2 fails (some driven checkpoint succeeded, but rule B's picks did
  not clear the bar of 7.4), the balanced criterion is reported as a warning, not as a way to
  choose. If rule B picks rule A's checkpoint in all six runs, H2 is reported as "not informative:
  same picks in 6 of 6 runs"; otherwise, if no driven checkpoint succeeds in any episode (row 5), as
  "not informative: no success to rank", not as failed. In every case the number of runs in which
  the two rules pick the same checkpoint is given. If the offline step could not be used (7.4), H2
  is reported as not run, with the reason;
- rules C, D and G are reported as description, their p-values labelled "unadjusted,
  descriptive"; if one's E exceeds rule A's, the paper says so without calling it a test;
- the descriptions of section 8 are reported in every row: each run's selection regret, labelled as
  biased upward, and the number of episodes any of the 37 policies solved;
- the comparator episodes that H1 and confirmation counted as successes because they were not
  completed are listed beside H1 (section 5);
- H3: if it holds, one sentence says that the offline STOP count predicted no closed-loop STOP for
  the 15 unselected unweighted checkpoints, as it had for the 3 selected ones on the Spark, with the
  number of completed episodes if any are missing; if it fails, the paper reports where;
- the cross-check of 7.8 is reported with its fixed wording;
- an aborted run, if the run moved (4.2), is reported beside the result;
- the abstract's "The proxy gave no warning" is limited to the averaged CE (review item Z6).

## 10. What will not be concluded

1. Anything about `val_unseen` or buildings not seen in training. `val_unseen` stays sealed.
2. Anything about other backbones, learning rates, budgets, data sizes or objectives, or about the
   single-seed probes (chunk objective, 40k, 134k, geometry).
3. That teacher-forced CE is a poor selector in general. Even outcome 1 is one case.
4. That a checkpoint below its H1 bar is better, even one success short. One success is a fact about
   one episode, not a rate.
5. A ranking of the 31, or a "best checkpoint", from the main suite alone.
6. From outcome 5, that fine-tuning cannot teach stopping, or that VLM policies cannot do R2R-CE.
   Outcome 5 supports only this: none of these 37 policies, at this frozen budget (192 updates,
   3,072 decisions, rank-32 LoRA, learning rate 2e-5, these three seeds), stops inside the goal
   radius on these 61 episodes on the study GPU.
7. Anything from differences in oracle success, navigation error or nDTW about which checkpoint is
   better.
8. That a criterion in section 8 predicts success, unless its cell clears 0.0025 (0.00625 in the
   fallback of 7.4). An untestable cell says nothing either way. No cell supports a causal claim.
9. That rule B is a remedy in general, even if H2 passes; and that rule C, D or G selects better,
   from their description alone. If the offline step could not be used, anything about rule B.
10. Any ordering of weighted against unweighted training. This experiment is not paired for that.
11. Anything about other GPU types. Every closed-loop number and every offline criterion comes from
    one type, in one image; the cross-check of 7.8 cannot separate the GPU from the renderer or the
    code, and the scorer check of 7.4 is a check of the port, not a comparison of machines.
12. That a rollout on the study GPU is a fixed function of its checkpoint, if the smoke showed that
    sharing a GPU, or a resume, changes a rollout.

## 11. Compute

Nothing runs on the Spark's GPU. Nothing has been measured for this model on any GPU of the cluster;
the smoke measures it, and its results give the projection before the freeze commit.

| step | GPU hours | basis |
|---|---|---|
| smoke | at most about 3 | at the GB10's rate, 8 episodes alone and 4 x 8 shared are at most 40 x 500 x 0.23 s = 1.3 h, plus phase 0's 1,152 decisions (about 0.1 h), phase 3's four scorers at once (at most about 0.4 h), eleven model loads, and phase 4's 244 control runs without a model, run first, with about 87 scene loads (an estimate of 15 to 30 minutes, not measured); the job's limit is 3 h, and every phase's processes stop at an episode boundary before it, so that the smoke reports even when it runs out of time |
| offline | at most about 5 | 37 x 1,152 decisions, about 3.5 h at the GB10's rate (the fourth draft's estimate for the same scoring), plus 37 model loads (about 0.9 h) and up to about 0.5 h lost to preemption |
| main | at most about 82 | 37 x 2.073 h = 76.7 h if a worker on the study GPU were no faster than the GB10 and P workers sharing a GPU went no faster in total than one, plus 37 model loads (about 0.9 h) and up to about 4 h lost to preemption |
| confirmation | at most about 20 | at most nine drives of 2.073 h (three candidates and, if update 0 is carried, all six selected checkpoints), plus loads and up to about 1 h lost to preemption; run only if a candidate clears the H1 bar |

The ceilings rest on an assumption nothing has tested: that a worker on the study GPU is no slower
than one on the GB10. The offline step and main use the same P, which section 6's remedy may raise
above 4 before the freeze. Every type the cluster offers has
several times the GB10's memory bandwidth, and P workers share each GPU, so the real figure should
be far lower; the CPU work per decision (rendering, image preparation) limits the gain by an amount
nobody here has measured. With P = 4, the main run needs ten one-GPU jobs, within the per-user limit
of 16 GPUs for jobs under two days; with fewer workers per GPU it needs more jobs, run in waves
beyond 16. Queue time is unknown. The author approves the ceilings before the freeze commit **[!]**,
so that no approval is needed after a result is known. The 37 rollouts take about 6.0 GB.

## 12. What this version gives up, and other departures

**Given up against the fourth draft, plainly:**
- **The offline scores come from the study GPU, not the GB10 that made the training records.** The
  fourth draft scored on the Spark and required near bit-identity (no token NLL more than 0.25 nats
  off). This draft scores on the study GPU so that the criteria and the rollouts share one GPU type,
  and its scorer check (7.4) tolerates hardware rounding; it would let through a scorer that shifts
  every value by less than 0.05 nats or flips only decisions won by less than a nat. There is no
  scorer fix after the freeze commit, and no re-run; the smoke's phase 0 is the chance to find a
  broken port.
- **Order and uniqueness of runs rest on git and on the author.** Git records what was committed; it
  cannot show that a run directory on the cluster was not deleted and started again before its
  results came back, or that a smoke was not repeated until it looked good. Commit times are set by
  the committer; only the freeze commit can get an outside time, from a pushed tag. The fourth
  draft's refusals between steps are gone.
- **No independent verifier program.** The analysis script checks the episode reports, and a reader
  who did not write it re-derives the numbers by hand.
- **No guard in the programs against two workers on one checkpoint.** The kit's claims (one
  directory per checkpoint, made atomically) and Slurm's `singleton` dependency per job name are the
  only protection.

**Departures logged now.**
- The C0 closed-loop protocol drove only selected checkpoints and set its scope with a latency gate;
  this experiment sets its scope in this file.
- Rule A is the six recorded picks.
- The comparators are the six selected checkpoints driven again on the study GPU; their Spark
  recordings are a cross-check that decides nothing.
- H1 and confirmation use the general McNemar p, because a comparator driven again may succeed; with
  the comparator at 0, the confirmation bars are those of the earlier drafts. The H1 screen of 7.3
  replaces the earlier drafts' Bonferroni over 31, whose bar of 10 it uses only when K is 26 or
  more.
- The decision page (`docs/STOPBLIND_ESR_DECISION_2026-09-30.md`) said "with the same runner"; the
  runner is extended with new code. Its bars of 5 and 10 hold only for the one-sided exact McNemar
  test with the comparator at 0; under the screen of 7.3 the H1 bar with the comparator at 0 is 5 to
  10. It dropped the confirmation that `D4_final.md` required; this file restores it. Its "about 65
  GPU-hours" on the Spark becomes a ceiling of about 3 + 5 + 82 plus 20 GPU-hours on the study GPU.
- `D4_final.md` counted rule B's successes over the 366 run-episodes; H2 counts distinct episodes,
  so that one easy episode solved by several picks is not counted several times. The total over the
  366 run-episodes is still reported.
- The fifth draft's rule B_sel and criterion K7 (a class-balanced CE on the selection surface, whose
  STOP term rests on 3 decisions) are dropped: they stood in for rule B and K3, which the offline
  step restores. Their recorded values stay in the fifth draft
  (`docs/prereg_history/PREREG_31_CHECKPOINTS.draft5.md`) and in the revision log.
- The smoke's phase 4 restores the fourth draft's replay of the published controls before the
  freeze commit, with the fourth draft's bar (one forward step, 0.25 m) on the main suite's final
  distances, and adds a check that every confirmation episode loads. The fourth draft also required
  the confirmation replays to succeed; this draft reports them and does not require them.

## 13. What this draft could not check

- Anything on the cluster: the driver, whether the image builds and passes its test there, whether
  its EGL renders through NVIDIA, which GPU types the account reaches, the speed, the memory per
  worker, whether P workers keep the GPU above the use below which the cluster cancels jobs, queue
  times and preemptions. The smoke answers most of these before the freeze commit, and its phase 4
  shows whether the study GPU's simulator reproduces the Spark's control runs on the 61 main
  episodes and runs all 61 confirmation episodes.
- Whether a rollout on the study GPU is reproducible across workers, resumes and nodes; the smoke
  measures it for one checkpoint on eight episodes.
- Whether the study GPU renders the same pixels as the GB10. No frame was kept on the Spark.
- Whether the study GPU's offline scores agree with the GB10's records within the scorer check's
  bars; the smoke's phase 0 measures it for wt25-u192, on its 1,152 recorded decisions.
- Whether the snapshot's non-weight files are the ones training read: no run recorded their hashes.
- Whether the study GPU's nodes all run one driver; a job starts nothing on a node whose driver is
  not the smoke's, and the analysis refuses results from two (4.2).
- The adapter's dtype inside the loaded model: training saved FP32 tensors, and the loader copies
  each into the parameter's own dtype; the dtype check after every load answers it on the cluster.

## Appendix A: evidence checked for this draft (CPU only)

- **The six `result.json` files** hash as `native_c0_closed_loop_v1` pinned them (6 of 6, checked
  2026-10-04):

| tag | `result.json` | sha256 |
|---|---|---|
| uw25 | `native_c0_development_v1/result.json` | `88bbd74f41eb6bb9d673e12fb3aa094accc5b4a93e5e6c0fe81c8352a3339c61` |
| wt25 | `native_c0_action_balance_v1/result.json` | `ced3e7c213508bf0c80f1aeed60e732aaf71f2cd133eb056e716305a2a089866` |
| uw26 | `native_c0_seed_replication_unweighted_s20260826_v1/result.json` | `946d05b33a7dc860c287be6b4675507767693a4a13191b63cd6aff872b497c69` |
| wt26 | `native_c0_seed_replication_inverse_frequency_s20260826_v1/result.json` | `52ae0f71a5269ed92794b0412f87af9391ee25bff749b5c1dd553a18cc6f8aa0` |
| uw27 | `native_c0_seed_replication_unweighted_s20260827_v1/result.json` | `65082403f55eda913fc9967e3dc37c1f578e999353a9afd12490c1b25370e139` |
| wt27 | `native_c0_seed_replication_inverse_frequency_s20260827_v1/result.json` | `46fe665925d94fd4070ee94d97b0b1e7657a214d78d4b9c2e0f8d6f2ebb2978c` |

- **What each records.** `calibrations` holds seven entries, updates 0, 32, 64, 96, 128, 160 and
  192, each with a `summary` of the 256 selection decisions (mean CE, the predicted-action
  histogram, and CE and recall per target action) and 256 per-decision records. `primary_evaluation`
  (the 640 natural decisions), `balanced_diagnostic` (the 256 balanced decisions) and
  `existence_gate` each hold one update only, the run's selected one (160, 192, 160, 128, 160 and
  192).
- **The selection surface** holds 173 FORWARD, 3 STOP, 42 LEFT and 38 RIGHT decisions in every run.
  K1 and K5 from these files equal the section 3 table.
- **The decisions the scorer check compares** (7.4), recounted on 2026-10-04 from the six files: the
  selection surface at the 37 policies (update 0 once) and the natural and balanced surfaces at the
  six selected ones, 37 x 256 + 6 x 896 = 14,848. Their recorded top-two margins: 551 at 0, 978 at
  0.125, 975 at 0.25, 3,773 from 0.375 to 0.875, and 8,571 at 1.0 nats or more.
- **The smoke episodes.** The release has 259 trajectories; 61 are in the main suite, 198 are
  eligible for confirmation, the confirmation draw takes 61 (reproduced on 2026-10-04: first ten
  4762, 4087, 6158, 7130, 2328, 6472, 3190, 6247, 5098, 6943; 34 scenes), and 137 remain. The four
  smallest are 93, 139, 149 and 180, each with three episodes; their two lowest episode ids are 4
  and 5, 10 and 11, 13 and 14, and 16 and 17.
- **McNemar bars,** the smallest b with P(Binomial(b + c, 1/2) >= b) at or below the level, for
  c = 0 to 4: 10, 13, 15, 17 and 19 at 0.05/31; 5, 7, 9, 10 and 12 at 0.05; 6, 8, 10, 12 and 13 at
  0.025; 6, 9, 11, 13 and 14 at 0.05/3. Rechecked 2026-10-08 in exact fractions.
- **The screen's bars** (7.3), at the threshold 0.05/K(0.05), for c = 0 to 4, recomputed 2026-10-08
  in exact fractions:

  | K(0.05) | c = 0 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|---|
  | 1 | 5 | 7 | 9 | 10 | 12 |
  | 2 | 6 | 8 | 10 | 12 | 13 |
  | 3 | 6 | 9 | 11 | 13 | 14 |
  | 4 | 7 | 9 | 11 | 13 | 15 |
  | 5 | 7 | 10 | 12 | 14 | 15 |
  | 6 | 7 | 10 | 12 | 14 | 16 |
  | 7 | 8 | 10 | 12 | 14 | 16 |
  | 8 | 8 | 10 | 13 | 15 | 16 |
  | 9-12 | 8 | 11 | 13 | 15 | 17 |
  | 13 | 9 | 11 | 13 | 15 | 17 |
  | 14-15 | 9 | 11 | 14 | 16 | 18 |
  | 16-22 | 9 | 12 | 14 | 16 | 18 |
  | 23 | 9 | 12 | 14 | 17 | 18 |
  | 24-25 | 9 | 12 | 15 | 17 | 19 |
  | 26-29 | 10 | 12 | 15 | 17 | 19 |
  | 30-31 | 10 | 13 | 15 | 17 | 19 |

- **Power of 7.7,** computed 2026-10-08 by exact enumeration (the 30 other candidates' counts as a
  multiset of their values of 5 or more, pruned below 1e-18, pruned mass 1.2e-14), for Bonferroni
  over 31, Tarone's procedure and its alpha-consistent form; the last two agree to four decimals in
  this setting, and the order of tied candidates in the carried list changes no figure at four
  decimals. The outside reading's simulations (4,000 replicates) gave 0.058 and 0.052 against 0.059
  for Bonferroni at 10%, and 0.514 and 0.526 against 0.523 for the screen.
- **O2's smallest p,** recomputed 2026-10-08: 2/720, 2/216, 2/1296 and 2/3375 for the four cases of
  section 8; and P(Binomial(10,000, 1/360) <= 24) = 0.273, the chance that draft 8's Monte Carlo p,
  (1 + extreme)/10,001, fell below 0.0025 for a one-run cell.
- **The family bound,** recomputed 2026-10-08: 1 - (0.05/37)^(1/61) = 0.1026 (Sidak's form, exact
  under independence, gives 0.1023); 1 - 0.05^(1/61) = 0.0479; 0.9^61 = 0.0016.
- **Speed on the GB10,** from the 366 recorded episode reports: median seconds per decision
  0.2213-0.2286; decisions per checkpoint 28,023 to 30,500; model loads 86.2-92.9 s.
- **The image recipe:** `gridnav_torch.def` `bd386211...`, `versions.lock` `d66e5459...`; the
  versions in section 4.3 are those `versions.lock` pins and the image's test asserts.
- **The two suite files,** built on 2026-10-04 by `scripts/build_native_c0_confirmation_suite.py`
  from the release: `configs/stopblind/confirmation_suite.json` (sha256 `a34950cd...`; it reproduces
  the dry run) and `configs/stopblind/smoke_suite.json` (sha256 `bd4d681a...`; episodes 4, 5, 10,
  11, 13, 14, 16 and 17). The runner pins both hashes.
- **The scene check's inputs** (section 6, phase 4), checked 2026-10-08: the six files of the
  control suite that the check runs (`datasets.py`, `habitat_sim_r2r.py`, `habitat_suite.py`,
  `metrics.py`, `policy_ordering.py`, `r2r_ce.py`) hash as `policy_ordering_v1`'s manifest records
  them, and the two of them git did not track at `0a32cff` are `habitat_suite.py` and
  `policy_ordering.py`; that suite's 854 runs are 14 controls on 61 episodes; on the Spark, expert
  replay succeeded and stopped on 61 of 61 main episodes, in 37 to 111 steps, and always-forward
  stopped on none and reached the 500-step cap on all 61; the main suite is ordered by scene, its 53
  scenes each in one block, and the confirmation suite holds 34 scenes, so the check's 244 runs
  change scene 87 times, the first load included; the fourth draft's bar was a final distance
  within 0.25 m of the Spark's on the main replays, and it required the 61 confirmation replays to
  succeed (its check T, `docs/prereg_history/PREREG_31_CHECKPOINTS.draft4.md`).
- **The model snapshot list:** the 14 files of the Spark's snapshot, hashed on the workstation on
  2026-10-04 and committed in the runner (`MODEL_SNAPSHOT_FILES`); its five weight files equal what
  every training run recorded.
- **The content cache:** the 1,152 held-out objects the offline step opens take 0.70 GB of the
  cache's 2.56 GB (from the committed cache index).
- Checked for the earlier drafts (2026-09-30), and unchanged: all 42 checkpoint sha256 values (two
  independent re-hashes); the `lora_B` tensors of the six update-0 files all zero, and none zero in
  the 36 trained files; the 61 entries bound to the `val_seen` release by trajectory and scene id;
  the 366 recorded start positions equal to the release's.

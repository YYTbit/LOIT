# Execution and inputs

## Activity pool

```bash
python -m loit prepare-activities --output data/activities.jsonl \
  --manifest data/activities_manifest.json --seed 0
```

The default source is `databricks/databricks-dolly-15k`, distributed under CC-BY-SA 3.0. The command retains instructions and available context, removes source answers, maps creative writing to generation, filters benchmark names and unavailable-context requests, and deduplicates instructions. The manifest records the resulting category counts, seed, and content hash. `--input` accepts a local source JSON or JSONL file; specify its provenance and license with `--provenance` and `--license`. Activity records contain `activity_id`, `category`, and `instruction`.

## Scientific records

Every task carries an explicit `split` of `train`, `dev`, `validation`, or `test`, a `task_id`, and a `question`. Maintain disjoint instance identifiers across training, development, and evaluation manifests when preparing the scientific records. Optional fields are `observations`, `proposals`, `tools`, and prior `actions`. Reference answers and verification annotations stay in separate records. Residual memory encodes the question and observations.

Structured proposals contain five text fields, `hypothesis`, `competing_explanation`, `discriminating_test`, `prediction`, and `competing_prediction`. The last two fields specify the expected observation under each explanation. A prompt exceeding the configured context budget raises an error, allowing the input history to be shortened while retaining the question.

## Training

Each command saves a separate checkpoint. Start with scientific supervised fine-tuning in the README, then initialize the added modules.

```bash
python -m loit train --config configs/qwen3_8b.yaml --checkpoint checkpoints/focused \
  --stage initialize --records data/initialization.jsonl --output checkpoints/initialized

python -m loit collect-associations --config configs/qwen3_8b.yaml \
  --checkpoint checkpoints/initialized --tasks data/training_states.jsonl \
  --activities data/activities.jsonl --output data/association_rollouts.jsonl

python -m loit train --config configs/qwen3_8b.yaml --checkpoint checkpoints/initialized \
  --stage association --records data/verified_associations.jsonl --output checkpoints/association

python -m loit train --config configs/qwen3_8b.yaml --checkpoint checkpoints/association \
  --stage return --records data/verified_returns.jsonl --output checkpoints/return

python -m loit train --config configs/qwen3_8b.yaml --checkpoint checkpoints/return \
  --stage utility --records data/paired_training.jsonl --output checkpoints/utility

python -m loit train --config configs/qwen3_8b.yaml --checkpoint checkpoints/utility \
  --stage controller --records data/verified_episodes.jsonl --output checkpoints/controller
```

All optimization records require `split` set to `train`. Scientific supervision uses `state` and `response`. Initialization records contain `state`, with optional `activity`, `activity_response`, and independently verified `proposal` supervision. Verified return targets retain the corresponding `terminal_code`. Return supervision requires `state`, `terminal_code`, `proposal`, and `verified` set to `true`.

Association collection records each sampled response, proposed code, topic-change outcome, and behavior likelihood. Supply independent proposal feedback through `verified_reward` in [0, 1] before training. Evidence-gate scores remain separate from these labels. Memory and code representations are learned during initialization and retained during association learning and return refinement. Association replay uses recorded behavior likelihoods for importance weighting. Refresh trajectories as the association policy changes; a nonfinite importance weight or objective stops optimization.

Every collected output has a stable `record_id`. Independent label files retain that identifier and the same explicit split. Merge labels without changing trajectories using the following command. Controller labels contain `scientific_reward` and `idea_reward` and use `--stage controller`.

```bash
python -m loit attach-labels --stage association --records data/association_rollouts.jsonl \
  --labels data/independent_feedback.jsonl --output data/verified_associations.jsonl
```

Paired utility records contain `state_id`, `summary`, `option_index`, `incubation_return`, `focused_return`, and nonnegative `added_cost`. Match scientific states and continuation conditions. The summary is the recorded controller input. Returns combine normalized scientific progress and independently verified proposal quality, before the computation penalty. Added cost is measured in processed decoder positions divided by `cost_scale`.

Controller records are complete episodes with `complete_episode`, `events`, `processed_positions`, `incubation_positions`, `scientific_reward`, and `idea_reward`. Both rewards lie in [0, 1] and are supplied by independent verification. Total processed positions determine the episode budget; incubation positions determine the computation penalty. Events retain the summary, sampled option, admissible-action mask, and actor likelihood. Forced options support controlled interventions and are excluded from PPO updates. Episode returns are standardized using training records.

The configuration provides AdamW, cosine scheduling, gradient accumulation, and separate backbone and module learning rates. `configs/fsdp.yaml` provides an optional Accelerate launch configuration. Global batch size is the product of per-process batch size, accumulation, and process count. Select layer indices, attention heads, gate weights, generation limits, initial code distributions, and reward weights through the configuration. Unspecified layer indices use the configured depth fractions; unspecified initial code distributions are uniform. Gate thresholds and utility margins require development calibration.

## Development calibration

```bash
python -m loit calibrate --config configs/qwen3_8b.yaml \
  --checkpoint checkpoints/controller --gate-records data/gate_development.jsonl \
  --paired-records data/paired_development.jsonl --output data/calibration.json
```

Gate records contain a development split, `gate_score`, and an independent binary `verified` label. The selected threshold maximizes confirmation rate subject to the requested acceptance floor. Paired development records use the utility-training schema. The margin is the requested empirical quantile of absolute utility errors. These are development estimates. Deployment admits focused continuation and incubation options with utility strictly above the selected margin, then selects the actor's highest-probability admissible option.

## Benchmarks

[DiscoveryWorld](https://github.com/allenai/discoveryworld) and [ScienceWorld](https://github.com/allenai/ScienceWorld) use native SDK sessions. Install their released dependencies in a compatible environment, including Java for ScienceWorld. The manifest in `configs/environments.json` selects each factory. Set `sdk_python` in factory arguments to use a separate SDK interpreter.

DiscoveryWorld task parameters contain `scenario`, `difficulty`, and `world_seed`. ScienceWorld parameters contain `task_name`, `variation`, and optional `simplification`. An interactive task also specifies `benchmark`, its explicit split, and the scientific question.

```bash
python -m loit run --config configs/qwen3_8b.yaml \
  --checkpoint checkpoints/controller --calibration data/calibration.json \
  --tasks data/interactive_tasks.jsonl --activities data/activities.jsonl \
  --environments configs/environments.json --output results/episodes.jsonl
```

`--training-rollouts` collects stochastic controller actions on training tasks. PPO requires this sampling provenance. The output records complete episodes, native scores, and processed positions. Attach normalized scientific and independently verified idea rewards before controller training. `--position-budget` caps processed decoder positions, including initialization and screening. Budget exhaustion restores the saved scientific state when reintegration cannot complete. Oracle scorecards are excluded from agent observations.

For [TOMATO-Star](https://github.com/ZonglinY/MOOSE-Star), [LiveIdeaBench v2](https://github.com/x66ccff/liveideabench), and [AgentIdeaBench](https://github.com/HKUST-KnowComp/AgentIdeaBench), prepare questions using the released task definitions and generate answers with `propose`. Preserve the official task identifiers, required hypothesis counts, and complete evaluation collections. External evaluation is supplied as an importable `module:function` that accepts the prediction list and returns a dictionary of native metrics.

```bash
python -m loit score --benchmark liveideabench --predictions results/proposals.jsonl \
  --evaluator benchmark_interface:evaluate --output results/native_metrics.json
```

The evaluator is supplied by the benchmark installation. Collection-level metrics receive the complete prediction collection. In particular, retain LiveIdeaBench's native fluency procedure. No benchmark-specific substitute metric is applied.

## Transition analysis

`analyze` reads `initial`, `transition`, `quality`, `duration`, `transition_samples`, `return_samples`, and optional `delta` from JSON. Transition rows require independent controlled samples, and terminal-code quality requires separate independently verified return samples. It reports the finite-horizon useful-proposal probability and the paper's conservative finite-sample interval.

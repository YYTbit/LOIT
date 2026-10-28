# When to Incubate: Learning Off-Task Associations for Scientific Ideation

Scientific ideation requires unexpected associations, yet large language model
agents often revisit familiar hypotheses as observations become scarce.
Cognitive incubation suggests that unrelated activities may facilitate new
connections upon returning to a problem. Inspired by this process, we introduce
the Learned Off-Task Incubation Transformer (LOIT), which learns when to
incubate during scientific exploration. Specifically, LOIT preserves focused
reasoning and encodes the question and observations as residual memory. During
incubation, a separate Transformer stream processes off-task activities and
evolves associative codes through memory-guided transitions and stochastic topic
changes. Return attention then integrates the final code into screened
proposals, while a reinforcement learning controller selects incubation actions
using predicted gain and cost. To understand how incubation affects proposal
quality, we model associative dynamics as a finite-state Markov process and
derive a convergence bound and a sufficient condition for improvement. Across
five benchmarks and three backbones, LOIT achieves the highest mean score in all
15 settings, including matched controls. For example, on Qwen3-8B,
DiscoveryWorld normalized score improves from 13.0% to 22.0%, and ScienceWorld
reward from 19.2% to 33.7%. Code is available at
<https://anonymous.4open.science/r/LOIT-1700/>.

[![Task-focused reasoning and LOIT](assets/overview.png)](assets/overview.pdf)

## Installation

Use Python 3.10 or later and a PyTorch build compatible with the execution
hardware.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,data,test]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

## Data preparation

Prepare scientific records with explicit training, development, and evaluation
splits. Each task contains a question, observations, and a stable identifier.
Independently verified proposal and continuation labels use the schemas in
[docs/execution.md](docs/execution.md).

Build the off-task activity pool from Dolly instructions.

```bash
python -m loit prepare-activities --output data/activities.jsonl \
  --manifest data/activities_manifest.json --seed 0
```

The collector removes source answers, filters benchmark references, and
deduplicates activities. The output preserves source and license metadata.

## Training

Train focused reasoning, initialize the added modules, and fit the association,
return, utility, and controller stages in sequence. Each stage consumes the
corresponding records described in the execution guide.

```bash
python -m loit train --config configs/qwen3_8b.yaml --stage focused \
  --records data/scientific_training.jsonl --output checkpoints/focused

python -m loit train --config configs/qwen3_8b.yaml --stage initialize \
  --checkpoint checkpoints/focused --records data/initialization.jsonl \
  --output checkpoints/initialized

python -m loit train --config configs/qwen3_8b.yaml --stage association \
  --checkpoint checkpoints/initialized --records data/verified_associations.jsonl \
  --output checkpoints/association

python -m loit train --config configs/qwen3_8b.yaml --stage return \
  --checkpoint checkpoints/association --records data/verified_returns.jsonl \
  --output checkpoints/return

python -m loit train --config configs/qwen3_8b.yaml --stage utility \
  --checkpoint checkpoints/return --records data/paired_training.jsonl \
  --output checkpoints/utility

python -m loit train --config configs/qwen3_8b.yaml --stage controller \
  --checkpoint checkpoints/utility --records data/verified_episodes.jsonl \
  --output checkpoints/controller
```

`collect-associations` generates trajectories with recorded behavior
likelihoods. Association and controller training consume independently supplied
verification labels. `configs/fsdp.yaml` provides an optional Accelerate
configuration.

## Development calibration

Calibrate the proposal gate and utility margin on designated development
records.

```bash
python -m loit calibrate --config configs/qwen3_8b.yaml \
  --checkpoint checkpoints/controller --gate-records data/gate_development.jsonl \
  --paired-records data/paired_development.jsonl --output data/calibration.json
```

Deployment reads these thresholds when screening proposals and selecting
admissible incubation actions.

## Inference and evaluation

Generate structured hypotheses and discriminating tests from scientific
questions.

```bash
python -m loit propose --config configs/qwen3_8b.yaml \
  --checkpoint checkpoints/controller --calibration data/calibration.json \
  --tasks data/tasks.jsonl --activities data/activities.jsonl \
  --output results/proposals.jsonl
```

DiscoveryWorld and ScienceWorld use `run` with the native factories in
[configs/environments.json](configs/environments.json). TOMATO-Star,
LiveIdeaBench v2, and AgentIdeaBench use prepared task records and their native
evaluators. The [benchmark guide](docs/execution.md#benchmarks) describes SDK
preparation, collection-level scoring, and the AgentIdeaBench Active entry
point.

## Configuration

Profiles are provided for [Qwen3-8B](configs/qwen3_8b.yaml),
[Llama-3.1-8B-Instruct](configs/llama_3_1_8b.yaml), and
[DeepSeek-R1-Distill-Qwen-7B](configs/deepseek_r1_qwen_7b.yaml).

| Setting | Value |
| --- | --- |
| Residual-memory slots | 4 |
| Associative codes / code slots | 8 / 4 |
| Topic-change probability | 0.24 |
| Incubation durations | 2, 4, 6, 8, 12 |
| Coupling grid | 0, 0.25, 0.5, 0.75, 1 |
| Proposals per question | 4 |

Configure seeds, generation limits, reward weights, and training settings in the
selected profile. `analyze` computes finite-state proposal probabilities and
sample-based intervals from supplied transition and return records.

## Software checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The CPU suite checks memory and code transitions, controller updates,
calibration, token accounting, and checkpoint restoration.

## License

[MIT License](LICENSE).

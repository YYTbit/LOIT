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
reward from 19.2% to 33.7%.

## Algorithm Flow

Scientific ideation can stall when continued reasoning revisits familiar
hypotheses. Inspired by cognitive incubation, LOIT temporarily shifts
computation to unrelated activities while preserving scientific evidence. First,
residual memory conditions the associative transitions. Next, return attention
converts the terminal state into a testable proposal. Finally, a learned
controller allocates incubation according to expected scientific gain and cost.

[![Task-focused reasoning and LOIT](assets/overview.png)](assets/overview.pdf)

**Figure 1.** Task-focused reasoning and LOIT on DiscoveryWorld’s Plant Nutrients Challenge. In seed 1, growth requires low titanium together with medium barium. The three observations challenge a single-factor explanation. A color-mixing instruction from Dolly 15k provides the off-task input. Memory cross-attention retains the scientific observations, the transition head updates an associative code, and return cross-attention carries this code into a proposal for separate and joint nutrient tests.

**Problem Formulation.**

We model discovery as a partially observed decision process. At turn $`t`$, history $`x_t=(Q_t,E_t,F_t,\mathcal T_t)`$ contains the question, observations, hypotheses, and tools. A large language model with parameters $`\omega`$ selects action $`a_t`$ and receives an observation that updates $`x_{t+1}`$. The benchmark scores the resulting trajectory $`\tau`$.

To allocate computation when progress stalls, an incubation controller reads

```math
\begin{aligned}
u_t=[n_t,\rho_t,b_t,\operatorname{Mean}(M_t)],
\end{aligned}
\tag{1}
```

where $`n_t`$ measures observation novelty, $`\rho_t`$ hypothesis and action repetition, $`b_t`$ the remaining budget fraction, and $`\operatorname{Mean}(M_t)`$ the mean of the residual-memory slots defined below. These features use the agent-visible prefix.

The controller chooses from $`\mathcal O=\{0\}\cup(\mathcal K\times\mathcal A\times\mathcal D)`$, where $`0`$ denotes focused continuation, $`\mathcal K`$ durations, $`\mathcal A\subseteq[0,1]`$ coupling strengths, and $`\mathcal D`$ activity categories. Action $`o=(K,\alpha,d)`$ runs $`K`$ association steps at coupling $`\alpha`$ in category $`d`$, preserving the history and environment. A development-selected rate $`\varepsilon\in(0,1)`$ controls topic changes.

We optimize the episode return

```math
\begin{aligned}
R(\tau)=R_{\mathrm{sci}}(\tau)+\lambda_I R_{\mathrm{idea}}(\tau)
-\beta C_{\mathrm{inc}}(\tau),
\end{aligned}
\tag{2}
```

where $`R_{\mathrm{sci}}`$ measures task progress, $`R_{\mathrm{idea}}`$ training-time proposal quality, and $`C_{\mathrm{inc}}`$ incubation computation. Nonnegative weights $`\lambda_I`$ and $`\beta`$ balance quality and cost. Final evaluation uses benchmark outcomes and independent proposal assessment.

**Residual memory.**

To preserve evidence while limiting repeated hypothesis narratives, the deterministic operator $`\operatorname{Canon}`$ serializes the question and observations as $`B_t=\operatorname{Canon}(Q_t,E_t)`$. A trainable resampler $`C_\psi`$, with parameters $`\psi`$, compresses backbone token representations $`H_\omega`$ into

```math
\begin{aligned}
M_t=C_\psi(H_\omega(B_t))\in\mathbb R^{m\times D},
\end{aligned}
\tag{3}
```

with $`m`$ memory slots and hidden width $`D`$. Previous hypotheses $`F_t`$ remain in the focused history, while the off-task stream receives scientific context through the question and observed evidence in $`M_t`$.

**Associative transitions.**

To carry associations across activities, a learned codebook $`\mathcal E=\{E_1,\ldots,E_N\}`$ represents $`N`$ states with $`r`$ vectors per code, $`E_i\in\mathbb R^{r\times D}`$. State $`z_k`$ indexes $`E_{z_k}`$. The initial code follows $`z_0\sim\nu_d`$, where $`\nu_d`$ is a row distribution over codes. Each step draws activity $`\xi_k\sim\mathcal P_d`$, where $`\mathcal P_d`$ is the category-specific distribution over Dolly 15k instructions. The controller selects category $`d`$; subsequent instruction draws are independent of the scientific question given that category.

The off-task stream processes $`\xi_k`$ and $`E_{z_k}`$ using the focused stream's Transformer weights and a separate cache. At selected layers $`\mathcal L_o`$, memory cross-attention modifies the shared block as

```math
\begin{aligned}
\widetilde H^o_{k,\ell+1}&=T^\omega_\ell(H^o_{k,\ell};\mathcal C^o_k),\\
H^o_{k,\ell+1}&=\widetilde H^o_{k,\ell+1}
+\alpha\,\mathrm{CA}_{\phi,\ell}(\widetilde H^o_{k,\ell+1},M_t).
\end{aligned}
\tag{4}
```

Here $`T^\omega_\ell`$ is Transformer block $`\ell`$, $`H^o_{k,\ell}`$ its off-task state, and $`\mathcal C^o_k`$ the activity-local causal cache. The association parameters $`\phi`$ include cross-attention and the transition head below. Other layers retain the shared Transformer update. The stream generates activity response $`y_k`$ and proposes the next code through

```math
\begin{aligned}
\widetilde z_{k+1}\sim\operatorname{Cat}\!\left(
\operatorname{softmax}(W_z\operatorname{Pool}(H^o_{k,L})+b_z)\right),
\end{aligned}
\tag{5}
```

Here $`\operatorname{Cat}`$ denotes categorical sampling, $`L`$ the backbone depth, $`\operatorname{Pool}`$ response-state pooling, and $`W_z,b_z`$ the transition-head parameters. Memory supplies cross-attention keys and values; off-task states supply queries.

**Topic changes.**

To permit movement across associations, a step retains $`\widetilde z_{k+1}`$ with probability $`1-\varepsilon`$ or samples from $`\nu_d`$ with probability $`\varepsilon`$. Releasing the activity-local cache leaves the code as the persistent state. Conditional on $`(M_t,\alpha,d)`$, its transition matrix is

```math
\begin{aligned}
\mathbf P_{\alpha,M_t,d}=(1-\varepsilon)\widetilde{\mathbf P}_{\alpha,M_t,d}
+\varepsilon\mathbf1\nu_d.
\end{aligned}
\tag{6}
```

Here $`\widetilde{\mathbf P}(i,j)`$ averages the probability of proposing $`j`$ from $`i`$ over activities and responses, and $`\mathbf1`$ is an $`N`$-dimensional column of ones. This model supports transition estimation and state interventions.

**Insight Reintegration.**

To convert the association into a scientific test, the return branch resumes the saved focused cache $`\mathcal C^f_t`$ and attends to $`E_{z_K}`$ at selected layers.

```math
\begin{aligned}
\widetilde H^r_{\ell+1}&=T^\omega_\ell(H^r_\ell;\mathcal C^f_t),\\
H^r_{\ell+1}&=\widetilde H^r_{\ell+1}
+\gamma_\ell\mathrm{CA}^{R}_{\eta,\ell}(\widetilde H^r_{\ell+1},E_{z_K}),
\end{aligned}
\tag{7}
```

Here $`H^r_\ell`$ is the return state, $`\eta`$ the attention parameters, and $`\gamma_\ell\in[0,1]`$ its strength. Each of up to $`J`$ proposals specifies a hypothesis, rival explanation, discriminative test, and predictions. The evidence gate $`G(s;x_t)`$ scores candidate $`s`$ for relevance, evidence consistency, novelty against $`F_t`$, and test quality. A development-selected threshold $`\tau_g`$ defines admissibility through $`G(s;x_t)\geq\tau_g`$. The highest-scoring admissible proposal enters the history, and the focused cache is rebuilt. An empty admissible set restores the saved history and cache.

Let $`Y_x(s)\in\{0,1\}`$ indicate independent verification of proposal $`s`$ as novel, consistent, and testable at state $`x`$. Define

```math
\begin{aligned}
w_x(j)=\Pr(Y_x(s)=1\mid x,z_K=j).
\end{aligned}
\tag{8}
```

This probability includes generation and screening; rejected and empty outcomes receive label zero. The return branch conditions on the saved history and terminal code, while coupling affects the preceding transitions. Gate acceptance and independent verification are recorded separately.

**Training and Adaptive Incubation.**

To align incubation with scientific utility, training first adapts the shared backbone through scientific supervised fine-tuning. Subsequent stages initialize memory, association, and return modules, optimize proposal generation, fit the utility estimator, and train the controller.

**Association learning.**

To learn useful transitions, the memory resampler first learns to preserve question and observation information. Supervised activity generation and return prediction initialize the association and reintegration modules. Verified rewards then train the sampled transitions through

```math
\begin{aligned}
\mathcal L_{\mathrm{assoc}}=-\mathbb E\left[
(r_x(s)-b_\chi(x))\sum_{k=0}^{K-1}\ell_{\phi,k}\right],
\end{aligned}
\tag{9}
```

where $`r_x(s)\in[0,1]`$ is verified reward, $`b_\chi`$ a reward baseline with parameters $`\chi`$, and $`\ell_{\phi,k}=\log p_\phi(y_k,\widetilde z_{k+1}\mid z_k,\xi_k,M_t,\alpha)`$ the response-and-transition log probability. The expectation covers sampled trajectories. This score-function objective assigns terminal proposal feedback to the preceding associative transitions. Successful proposals $`s^+`$ train the return decoder through $`-\log p_\eta(s^+\mid x,E_{z_K})`$.

**Paired utility estimation.**

To allocate computation by scientific benefit, training compares returns $`r_t^I(o)`$ and $`r_t^F`$ from incubation and matched focused continuation at the same state. Both combine scientific progress and proposal quality. Their difference is $`\Delta_t(o)=r_t^I(o)-r_t^F`$, with added cost $`C_t(o)`$. A model with parameters $`\vartheta`$ predicts gain $`\widehat D_\vartheta(u_t,o)`$ and cost $`\widehat C_\vartheta(u_t,o)`$ by squared-error regression. Predicted utility is $`\widehat U_\vartheta=\widehat D_\vartheta-\beta\widehat C_\vartheta`$, with $`\widehat U_\vartheta(u_t,0)=0`$.

**Policy optimization and execution.**

Proximal policy optimization trains actor $`\pi_\theta(o\mid u_t)`$ on Eq. (2), with critic $`V_\kappa(u_t)`$ estimating episode return. Their parameters are $`\theta`$ and $`\kappa`$, respectively. At deployment, focused continuation and incubation actions with predicted utility above a validation-selected margin $`m_v\geq0`$ are admissible.
The actor selects its highest-probability admissible action, with ties favoring focused continuation. The utility estimator therefore determines admissibility, while the actor supplies learned action preferences. Memory is constructed before action selection. Incubation then preserves the focused state, runs $`K`$ transitions, and screens proposals before interaction resumes (Algorithm 1).

**Finite-State Analysis of Incubation.**

To interpret incubation duration, condition on $`(x,M_t,d)`$ and write $`\mathbf P_\alpha`$ for the kernel and $`\nu`$ for the initial distribution. Averaging terminal usefulness $`w_x`$ after $`K`$ transitions gives

```math
\begin{aligned}
q_x(K,\alpha,d)=\nu\mathbf P_\alpha^K w_x.
\end{aligned}
\tag{10}
```

Let $`\mu_\alpha`$ be the stationary distribution and $`q_\infty=\mu_\alpha w_x`$. Topic changes contract total variation by $`1-\varepsilon`$ per step, yielding

```math
\begin{aligned}
|q_x(K,\alpha,d)-q_\infty|\leq(1-\varepsilon)^K.
\end{aligned}
\tag{11}
```

Thus $`q_\infty-(1-\varepsilon)^K>q_x^F(K)`$ suffices to exceed the useful-proposal probability $`q_x^F(K)`$ of compute-matched focused continuation. This criterion connects mixing speed with the usefulness of reached states.

![Algorithm 1](assets/algorithm.png)

## Results

### Main Results

**Table 1.** Main results on five public benchmarks, reported as mean ± standard deviation over five seeds. DW reports the released normalized score and SW the released normalized reward, both in percent, TS a 0–12 rubric, LIB v2 native fluency, and AIB the Active critic score.

![Main results from Table 1](assets/main_results.png)

**LOIT improves both interactive discovery and scientific ideation.** Relative to focused SFT, ScienceWorld reward increases by 14.5, 14.1, and 14.8 points across the three backbones. The DiscoveryWorld normalized score improves by 9.0, 7.8, and 8.7 points. Gains also extend to the three proposal-generation benchmarks. These gains suggest that returned associations support hypothesis generation and subsequent experiment selection.

**MOOSE-Star remains competitive on hypothesis composition.** MOOSE-Star reaches 5.13 on TOMATO-Star with Qwen3-8B and DeepSeek, compared with LOIT’s 5.41 and 5.39. Its performance on hypothesis composition is consistent with its scientific training objective. LOIT also improves the interactive scores, where a proposal guides evidence collection over subsequent actions. The comparison therefore highlights the role of connecting associative exploration to executable tests across task formats.

## Getting Started

### Environment Setup

```bash
conda create -n loit python=3.10 -y
conda activate loit
pip install -e '.[models,data]'
```

### Data Preparation

The experiments use [DiscoveryWorld](https://github.com/allenai/discoveryworld),
[ScienceWorld](https://github.com/allenai/ScienceWorld),
[TOMATO-Star](https://github.com/ZonglinY/MOOSE-Star),
[LiveIdeaBench v2](https://github.com/x66ccff/liveideabench), and
[AgentIdeaBench](https://github.com/HKUST-KnowComp/AgentIdeaBench).

Prepare the permitted scientific tasks and independently verified training
records using the [record formats](docs/execution.md#scientific-records).
Keep model-dependent rollouts and labels under `data/<model>/`. Build the shared
off-task activity pool:

```bash
python -m loit prepare-activities --output data/activities.jsonl \
  --manifest data/activities_manifest.json
```

## Training

Select a backbone:

- **Qwen3-8B**: `MODEL=qwen3_8b`
- **Llama-3.1-8B-Instruct**: `MODEL=llama_3_1_8b`
- **DeepSeek-R1-Distill-Qwen-7B**: `MODEL=deepseek_r1_qwen_7b`

Train the focused model, initialize the added modules, and continue through
association, return, utility, and controller training:

```bash
python -m loit train --config "configs/${MODEL}.yaml" --stage focused \
  --records data/scientific_training.jsonl --output "checkpoints/${MODEL}/focused"

python -m loit train --config "configs/${MODEL}.yaml" --checkpoint "checkpoints/${MODEL}/focused" \
  --stage initialize --records "data/${MODEL}/initialization.jsonl" --output "checkpoints/${MODEL}/initialized"

python -m loit train --config "configs/${MODEL}.yaml" --checkpoint "checkpoints/${MODEL}/initialized" \
  --stage association --records "data/${MODEL}/verified_associations.jsonl" --output "checkpoints/${MODEL}/association"

python -m loit train --config "configs/${MODEL}.yaml" --checkpoint "checkpoints/${MODEL}/association" \
  --stage return --records "data/${MODEL}/verified_returns.jsonl" --output "checkpoints/${MODEL}/return"

python -m loit train --config "configs/${MODEL}.yaml" --checkpoint "checkpoints/${MODEL}/return" \
  --stage utility --records "data/${MODEL}/paired_training.jsonl" --output "checkpoints/${MODEL}/utility"

python -m loit train --config "configs/${MODEL}.yaml" --checkpoint "checkpoints/${MODEL}/utility" \
  --stage controller --records "data/${MODEL}/verified_episodes.jsonl" --output "checkpoints/${MODEL}/controller"
```

Add `--resume` to the same command to continue interrupted training. For
distributed training, replace `python -m loit` with
`accelerate launch --config_file configs/fsdp.yaml -m loit`.

## Evaluation

Calibrate proposal thresholds on development records:

```bash
python -m loit calibrate --config "configs/${MODEL}.yaml" \
  --checkpoint "checkpoints/${MODEL}/controller" --gate-records "data/${MODEL}/gate_development.jsonl" \
  --paired-records "data/${MODEL}/paired_development.jsonl" --output "data/${MODEL}/calibration.json"
```

Generate proposals and run interactive tasks with the trained controller:

```bash
python -m loit propose --config "configs/${MODEL}.yaml" \
  --checkpoint "checkpoints/${MODEL}/controller" --calibration "data/${MODEL}/calibration.json" \
  --tasks data/tasks.jsonl --activities data/activities.jsonl \
  --output "results/${MODEL}/proposals.jsonl"

python -m loit run --config "configs/${MODEL}.yaml" \
  --checkpoint "checkpoints/${MODEL}/controller" --calibration "data/${MODEL}/calibration.json" \
  --tasks data/interactive_tasks.jsonl --activities data/activities.jsonl \
  --environments data/environments.json --output "results/${MODEL}/episodes.jsonl"
```

Add `--resume` to continue a prediction collection from its last completed task.

Configure the interactive environments and apply the benchmark scoring tools
following the [evaluation guide](docs/execution.md#environment-integration).

## License

[MIT License](LICENSE).

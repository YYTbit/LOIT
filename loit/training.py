from pathlib import Path
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from . import prompts
from .controller import paired_loss, ppo_loss
from .data import ScientificState, Proposal


class RecordDataset(Dataset):
    def __init__(self, path):
        self.path, self.offsets = Path(path), []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError("Training records are empty")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            record = json.loads(handle.readline())
        nested_split = record.get("state", {}).get("split")
        split = record.get("split", nested_split)
        if split != "train" or nested_split is not None and nested_split != split:
            raise ValueError("Optimization requires explicitly designated training records")
        return record


def episode_return(record, config):
    if not record.get("complete_episode") or not record.get("events"):
        raise ValueError("Controller updates require complete recorded episodes")
    scientific, idea = float(record["scientific_reward"]), float(record["idea_reward"])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in [scientific, idea]):
        raise ValueError("Scientific and independently verified idea rewards require [0, 1]")
    # The episode objective charges incubation computation; total positions still determine the budget.
    cost = float(record["incubation_positions"]) / config.cost_scale
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("The episode computation cost is invalid")
    return scientific + config.idea_weight * idea - config.cost_weight * cost


class Objectives(nn.Module):
    def __init__(self, language, stage, return_mean=0.0, return_std=1.0):
        super().__init__()
        self.model, self.language, self.stage = language.model, language, stage
        self.return_mean, self.return_std = return_mean, max(return_std, 1e-6)

    def _association(self, record):
        state = ScientificState.from_dict(record["state"])
        if record.get("verified_reward") is None:
            raise ValueError("Association learning requires independent proposal verification")
        reward = float(record["verified_reward"])
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError("Verified association rewards require [0, 1]")
        option = self.model.config.options[record["option_index"]]
        steps = record["steps"]
        if option.duration < 1 or len(steps) != option.duration:
            raise ValueError("The recorded trajectory length differs from its incubation option")
        if not math.isclose(record["topic_change"], self.model.config.topic_change) or not math.isclose(record["temperature"], self.model.config.temperature):
            raise ValueError("Recorded topic-change and sampling settings differ from the checkpoint")
        memory, _ = self.language.memory(state)
        summary = self.model.controller_state(memory, state.features())
        baseline = self.model.baseline(summary.detach()).squeeze()
        log_probability = torch.zeros((), device=self.model.device)
        code = record["initial_code"]
        if not 0 <= code < self.model.config.code_count:
            raise ValueError("The initial code lies outside the learned codebook")
        for step in steps:
            if step["previous_code"] != code or not 0 <= step["proposed_code"] < self.model.config.code_count or not 0 <= step["next_code"] < self.model.config.code_count:
                raise ValueError("The recorded code trajectory is inconsistent")
            if not step["reset"] and step["next_code"] != step["proposed_code"]:
                raise ValueError("A retained transition requires the proposed code")
            response_log_probability, logits = self.language.score_activity(memory, code, step["prompt_ids"], step["response_ids"], option.coupling)
            # The learned proposal receives credit even when the independent topic-change draw replaces it.
            log_probability = log_probability + response_log_probability + logits.log_softmax(-1)[step["proposed_code"]]
            code = step["next_code"]
        if code != record["terminal_code"]:
            raise ValueError("The terminal code differs from the recorded trajectory")
        if not math.isfinite(record["behavior_log_probability"]):
            raise ValueError("Recorded behavior likelihoods require finite values")
        # Behavior likelihoods permit replay while retaining the score-function expectation in Eq. (9).
        weight = (log_probability.detach() - record["behavior_log_probability"]).exp()
        if not torch.isfinite(weight):
            raise FloatingPointError("Refresh association rollouts before replaying this checkpoint")
        target = baseline.new_tensor(reward)
        return -weight * (target - baseline).detach() * log_probability + weight * F.mse_loss(baseline, target)

    def forward(self, records):
        if self.stage in ["utility", "controller"]:
            return self._heads(records)
        losses = []
        for record in records:
            if self.stage == "association":
                losses.append(self._association(record))
                continue
            state = ScientificState.from_dict(record["state"])
            if self.stage == "focused":
                losses.append(self.language.continuation_loss(prompts.focused(state), record["response"]))
            elif self.stage == "return":
                if record.get("verified") is not True:
                    raise ValueError("Return supervision requires independently verified proposals")
                proposal = Proposal(**record["proposal"])
                losses.append(self.language.return_loss(state, int(record["terminal_code"]), proposal.text()))
            elif self.stage == "initialize":
                memory, _ = self.language.memory(state)
                initial_code = int(record["initial_code"]) if "initial_code" in record else int(torch.randint(self.model.config.code_count, (), device=self.model.device))
                reconstruction = self.language.continuation_loss(
                    "Reconstruct the scientific question and observations from residual memory.",
                    state.canonical(), "association", memory, initial_code, 1.0)
                terms = [reconstruction]
                if "activity_response" in record:
                    terms.append(self.language.continuation_loss(prompts.activity(record["activity"]["instruction"]),
                                 record["activity_response"], "association", memory, initial_code, 1.0))
                if record.get("verified") is True and "proposal" in record:
                    proposal = Proposal(**record["proposal"])
                    terms.append(self.language.return_loss(state, int(record["terminal_code"]), proposal.text()))
                losses.append(torch.stack(terms).mean())
            else:
                raise ValueError("Unknown generative training stage")
        return torch.stack(losses).mean()

    def _heads(self, records):
        device = self.model.device
        if self.stage == "utility":
            for record in records:
                if not record.get("state_id"):
                    raise ValueError("Paired utility records require a shared scientific-state identifier")
                if not 0 < record["option_index"] < len(self.model.config.options):
                    raise ValueError("Paired utility records require an incubation option")
            summaries = torch.tensor([record["summary"] for record in records], device=device)
            actions = torch.tensor([record["option_index"] for record in records], device=device)
            gains = torch.tensor([record["incubation_return"] - record["focused_return"] for record in records], device=device)
            costs = torch.tensor([record["added_cost"] for record in records], device=device)
            if not torch.isfinite(summaries).all() or summaries.shape != (len(records), self.model.width + 3):
                raise ValueError("Paired utility records contain invalid controller summaries")
            if not torch.isfinite(gains).all() or not torch.isfinite(costs).all() or (costs < 0).any():
                raise ValueError("Paired gains and added costs require finite values")
            return paired_loss(self.model, summaries, actions, gains.float(), costs.float())
        events, returns = [], []
        for record in records:
            value = (episode_return(record, self.model.config) - self.return_mean) / self.return_std
            for event in record["events"]:
                if event.get("forced_option") or event.get("action_selection") != "sampled":
                    raise ValueError("PPO updates require sampled controller actions")
                events.append(event)
                returns.append(value)
        summaries = torch.tensor([event["summary"] for event in events], device=device)
        actions = torch.tensor([event["option_index"] for event in events], device=device)
        old = torch.tensor([event["actor_log_probability"] for event in events], device=device)
        masks = torch.tensor([event["admissible"] for event in events], dtype=torch.bool, device=device)
        if not torch.isfinite(summaries).all() or not torch.isfinite(old).all():
            raise ValueError("Controller rollouts require finite states and behavior probabilities")
        if (actions < 0).any() or (actions >= len(self.model.config.options)).any() or not masks.gather(1, actions[:, None]).all():
            raise ValueError("Recorded controller actions require admissibility under the behavior policy")
        return ppo_loss(self.model, summaries, actions, old, torch.tensor(returns, device=device), masks)


def train(language, stage, record_path, output, seed=0):
    from accelerate import Accelerator
    from .storage import save_checkpoint
    config = language.config
    torch.manual_seed(seed)
    dataset = RecordDataset(record_path)
    mean, std = 0.0, 1.0
    if stage == "controller":
        values = torch.tensor([episode_return(dataset[index], config) for index in range(len(dataset))])
        mean, std = float(values.mean()), float(values.std(unbiased=False).clamp_min(1e-6))
    language.model.select_trainable(stage)
    objective = Objectives(language, stage, mean, std)
    parameters = [parameter for parameter in objective.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The selected stage has no trainable parameters")
    learning_rate = config.backbone_learning_rate if stage == "focused" else config.learning_rate
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=tuple(config.adam_betas), weight_decay=config.weight_decay)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=list,
                        generator=torch.Generator().manual_seed(seed))
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation,
                              mixed_precision=config.precision if torch.cuda.is_available() else "no")
    objective, optimizer, loader = accelerator.prepare(objective, optimizer, loader)
    updates = max(1, math.ceil(len(loader) / config.gradient_accumulation) * config.epochs)
    warmup = int(updates * config.warmup_fraction)
    def rate(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        return config.minimum_lr_ratio + (1 - config.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    optimizer.zero_grad()
    for epoch in range(config.epochs):
        for records in loader:
            with accelerator.accumulate(objective):
                with accelerator.autocast():
                    loss = objective(records)
                if not torch.isfinite(loss):
                    raise FloatingPointError("The training objective is nonfinite")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(parameters, config.gradient_clip)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("The training gradient is nonfinite; refresh the recorded trajectories")
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
    state = accelerator.get_state_dict(objective)
    if accelerator.is_main_process:
        model_state = {key.removeprefix("model."): value for key, value in state.items()}
        save_checkpoint(language, output, model_state, {"stage": stage, "return_mean": mean, "return_std": std})
    accelerator.wait_for_everyone()

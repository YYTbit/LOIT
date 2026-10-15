from dataclasses import asdict, dataclass, field
from copy import deepcopy
import json
import math
import random

import torch

from . import prompts
from .controller import choose_action
from .data import Proposal, ScientificState
from .language import ComputationBudgetExceeded


@dataclass
class CostLedger:
    positions: dict[str, int] = field(default_factory=dict)
    limit: int | None = None

    @property
    def remaining(self):
        return None if self.limit is None else self.limit - self.total

    def require(self, count):
        if self.remaining is not None and count > self.remaining:
            raise ComputationBudgetExceeded("The remaining decoder-position budget is exhausted")

    def add(self, stage, count):
        if count < 0:
            raise ValueError("Computation counts require nonnegative values")
        self.require(count)
        self.positions[stage] = self.positions.get(stage, 0) + int(count)

    @property
    def total(self):
        return sum(self.positions.values())


@dataclass
class TurnResult:
    state: ScientificState
    cache: object
    event: dict


class Incubator:
    def __init__(self, language, activities):
        self.language, self.activities = language, activities
        self.model, self.config = language.model, language.config

    def _complete(self, costs, stage, *args, **kwargs):
        response = self.language.complete(*args, **kwargs, position_budget=costs.remaining)
        costs.add(stage, response.processed_tokens)
        return response

    def _screen(self, state, text, seed, costs):
        record = {"text": text, "valid": False, "gate_score": None, "verified_reward": None}
        try:
            candidate = Proposal(**json.loads(text))
        except (ValueError, TypeError):
            return record
        response = self._complete(costs, "screening", prompts.gate(state, candidate), self.config.gate_tokens, seed, schema=prompts.GATE_SCHEMA)
        try:
            scores = json.loads(response.text)
            values = [float(scores[name]) for name in prompts.GATE_SCHEMA["required"]]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                return record
        except (ValueError, KeyError, TypeError):
            return record
        record.update(valid=True, proposal=asdict(candidate), gate_components=scores,
                      gate_score=sum(weight * value for weight, value in zip(self.config.gate_weights, values)))
        return record

    @torch.no_grad()
    def turn(self, state, seed=0, cache=None, training=False, option_index=None, position_budget=None):
        if not training and (self.config.gate_threshold is None or self.config.utility_margin is None):
            raise ValueError("Load development-selected gate and utility calibration before deployment")
        rng = random.Random(seed)
        costs = CostLedger(limit=position_budget)
        if position_budget is not None:
            prefix_cost = len(self.language.ids(prompts.focused(state), reserve=self.config.proposal_tokens + 128)) if cache is None else 0
            memory_cost = len(self.language.ids(state.canonical(), reserve=self.config.code_slots))
            # Check both preparation passes before consuming any part of the remaining budget.
            costs.require(prefix_cost + memory_cost)
        if cache is None:
            cache = self.language.prefill(state)
            costs.add("focused_prefix", len(cache.prefix_ids))
        memory, memory_tokens = self.language.memory(state)
        costs.add("memory", memory_tokens)
        summary = self.model.controller_state(memory, state.features())
        selected, actor_log_probability, mask = choose_action(self.model, summary, training, rng.getrandbits(63), option_index)
        option = self.config.options[selected]
        event = {"state": state.to_dict(), "seed": seed, "summary": summary[0].cpu().tolist(),
                 "option_index": selected, "option": asdict(option), "actor_log_probability": actor_log_probability,
                 "forced_option": option_index is not None,
                 "action_selection": "forced" if option_index is not None else "sampled" if training else "argmax",
                 "admissible": mask.cpu().tolist(), "steps": [], "proposals": [], "selected_proposal": None,
                 "topic_change": self.config.topic_change, "temperature": self.config.temperature,
                 "completed_incubation": False, "budget_exhausted": False,
                 "behavior_log_probability": 0.0, "verified_reward": None}
        if option.duration == 0:
            event["costs"], event["processed_positions"] = costs.positions, costs.total
            event["incubation_positions"] = costs.total - costs.positions.get("focused_prefix", 0)
            return TurnResult(state, cache, event)
        try:
            updated, restored = self._incubate(state, cache, memory, option, seed, rng, event, costs)
        except ComputationBudgetExceeded:
            event["budget_exhausted"] = True
            updated, restored = state, cache
        event["costs"], event["processed_positions"] = costs.positions, costs.total
        event["incubation_positions"] = costs.total - costs.positions.get("focused_prefix", 0)
        return TurnResult(updated, restored, event)

    def _incubate(self, state, cache, memory, option, seed, rng, event, costs):
        category = self.config.categories.index(option.category)
        generator = torch.Generator(device=self.model.device).manual_seed(seed)
        code = int(torch.multinomial(self.model.initial[category], 1, generator=generator))
        event["initial_code"] = code
        for step in range(option.duration):
            activity = self.activities.sample(option.category, rng)
            activity_seed = rng.getrandbits(63)
            response = self._complete(costs, "activities", prompts.activity(activity.instruction), self.config.activity_tokens,
                                              activity_seed, "association", memory, code, option.coupling)
            pooling_cost = self.config.code_slots + len(response.prompt_ids) + len(response.token_ids)
            costs.require(pooling_cost)
            _, logits = self.language.score_activity(memory, code, response.prompt_ids, response.token_ids, option.coupling)
            costs.add("transition_pooling", pooling_cost)
            proposal = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            reset = rng.random() < self.config.topic_change
            next_code = int(torch.multinomial(self.model.initial[category], 1, generator=generator)) if reset else proposal
            transition_log_probability = float(logits.log_softmax(-1)[proposal])
            event["behavior_log_probability"] += response.log_probability + transition_log_probability
            event["steps"].append({"activity": asdict(activity), "seed": activity_seed, "previous_code": code,
                                   "prompt_ids": response.prompt_ids, "response_ids": response.token_ids,
                                   "response": response.text, "proposed_code": proposal,
                                   "reset": reset, "next_code": next_code})
            # Each response uses a fresh cache; the code index carries the persistent state.
            code = next_code
        event["terminal_code"] = code
        event["completed_incubation"] = True
        for attempt in range(self.config.proposal_count):
            try:
                response = self._complete(costs, "return", prompts.proposal(), self.config.proposal_tokens, rng.getrandbits(63),
                                         "return", code=code, cache=cache, schema=prompts.PROPOSAL_SCHEMA)
                event["proposals"].append(self._screen(state, response.text, rng.getrandbits(63), costs))
            except ComputationBudgetExceeded:
                event["budget_exhausted"] = True
                break
        admissible = [(index, record) for index, record in enumerate(event["proposals"])
                      if record["valid"] and self.config.gate_threshold is not None and record["gate_score"] >= self.config.gate_threshold]
        updated, restored = state, cache
        if admissible:
            index, best = max(admissible, key=lambda pair: pair[1]["gate_score"])
            updated = deepcopy(state)
            updated.proposals.append(best["proposal"])
            costs.require(len(self.language.ids(prompts.focused(updated), reserve=self.config.proposal_tokens + 128)))
            restored = self.language.prefill(updated)
            costs.add("rebuild", len(restored.prefix_ids))
            event["selected_proposal"] = index
        # Rejected or malformed candidates leave the saved scientific state and cache intact.
        return updated, restored


def run_episode(incubator, environment, task, seed=0, max_turns=32, position_budget=32768, training=False, option_index=None):
    if position_budget < 1 or max_turns < 1:
        raise ValueError("Episode budgets require positive values")
    session = environment.start(task, seed)
    events, total, incubation = [], 0, 0
    rng = random.Random(seed)
    try:
        state = ScientificState.from_dict({**task, "observations": [session.initial_observation],
                                           "tools": getattr(session, "tools", task.get("tools", []))})
        if state.split == "unspecified" or training and state.split != "train":
            raise ValueError("Episode collection requires an explicit compatible data split")
        for turn in range(max_turns):
            state.remaining_fraction = max(0.0, 1 - total / position_budget)
            if state.remaining_fraction == 0:
                break
            try:
                result = incubator.turn(state, rng.getrandbits(63), training=training, option_index=option_index,
                                        position_budget=position_budget - total)
            except ComputationBudgetExceeded:
                break
            state = result.state
            event = result.event
            total += event["processed_positions"]
            incubation += event["incubation_positions"]
            if total >= position_budget:
                events.append(event)
                break
            action = incubator.language.complete("", incubator.config.action_tokens, rng.getrandbits(63), cache=result.cache,
                                                 position_budget=position_budget - total)
            total += action.processed_tokens
            observation, done = session.step(action.text)
            state.actions.append(action.text)
            state.observations.append(observation)
            state.tools = getattr(session, "tools", state.tools)
            event["action"], event["observation"] = action.text, observation
            event["action_positions"] = action.processed_tokens
            events.append(event)
            if done:
                break
        score = float(session.score())
        if not math.isfinite(score):
            raise ValueError("The benchmark returned a nonfinite native score")
        return {"task_id": state.task_id, "split": state.split, "seed": seed, "complete_episode": True,
                "native_score": score, "events": events, "processed_positions": total,
                "incubation_positions": incubation,
                "cost_scale": incubator.config.cost_scale}
    finally:
        session.close()

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from loit.data import ActivityPool, ScientificState
from loit import prompts
from loit.language import Completion
from loit.runtime import Incubator, run_episode
from loit.training import Objectives


PROPOSAL = {"hypothesis": "H", "competing_explanation": "C", "discriminating_test": "T",
            "prediction": "P", "competing_prediction": "R"}


def instrument(language, monkeypatch, gate_score):
    complete = language.complete
    seen = []
    def generate(prompt, max_tokens, seed=0, branch="focused", memory=None, code=None, coupling=0, cache=None, schema=None, position_budget=None):
        seen.append((branch, cache))
        if branch == "association":
            return complete(prompt, max_tokens, seed, branch, memory, code, coupling, position_budget=position_budget)
        text = json.dumps(PROPOSAL) if branch == "return" else json.dumps(dict.fromkeys(["relevance", "consistency", "novelty", "test_quality"], gate_score))
        return Completion(text, [3], [4], 0.0, 1)
    monkeypatch.setattr(language, "complete", generate)
    pool = ActivityPool([{"activity_id": "a", "category": "brainstorming", "instruction": "Invent an object"}], language.config.categories)
    return Incubator(language, pool), seen


def test_rejected_return_restores_exact_state_and_cache(tiny, monkeypatch):
    language = tiny()
    incubator, seen = instrument(language, monkeypatch, 0.1)
    state = ScientificState("Q?", ["E"], split="train")
    original = asdict(state)
    cache = language.prefill(state)
    length = cache.values.get_seq_length()
    result = incubator.turn(state, 3, cache, training=True, option_index=1)
    assert result.state is state and result.cache is cache
    assert asdict(state) == original and cache.values.get_seq_length() == length
    assert result.event["selected_proposal"] is None
    assert all(item[1] is None for item in seen if item[0] == "association")
    assert all(item[1] is cache for item in seen if item[0] == "return")
    assert all(record["verified_reward"] is None for record in result.event["proposals"])


def test_accepted_return_appends_proposal_and_rebuilds_cache(tiny, monkeypatch):
    language = tiny()
    incubator, _ = instrument(language, monkeypatch, 0.9)
    state = ScientificState("Q?", ["E"], split="train")
    cache = language.prefill(state)
    result = incubator.turn(state, 5, cache, training=True, option_index=1)
    assert result.state is not state and state.proposals == []
    assert result.state.proposals == [PROPOSAL]
    assert result.cache is not cache and result.event["selected_proposal"] == 0
    assert result.event["costs"]["rebuild"] == len(result.cache.prefix_ids)


def test_independent_topic_changes_retain_proposed_code_credit(tiny, monkeypatch):
    language = tiny(topic_change=1, initial_distributions={"brainstorming": [1, 0, 0]})
    incubator, _ = instrument(language, monkeypatch, 0.1)
    state = ScientificState("Q?", ["E"], split="train")
    event = incubator.turn(state, 8, training=True, option_index=4).event
    assert len(event["steps"]) == 2 and event["terminal_code"] == 0
    assert all(step["reset"] and step["next_code"] == 0 for step in event["steps"])
    memory, _ = language.memory(state)
    expected = 0.0
    for step in event["steps"]:
        response, transition = language.score_activity(memory, step["previous_code"], step["prompt_ids"], step["response_ids"], event["option"]["coupling"])
        expected += float(response + transition.log_softmax(-1)[step["proposed_code"]])
    assert event["behavior_log_probability"] == pytest.approx(expected, abs=2e-5)
    event["verified_reward"] = 0.8
    language.model.select_trainable("association")
    loss = Objectives(language, "association")([event])
    assert torch.isfinite(loss)
    loss.backward()
    assert language.model.transition.weight.grad is not None
    assert language.model.transition.weight.grad.abs().sum() > 0


def test_deployment_requires_development_calibration(tiny, monkeypatch):
    language = tiny(gate_threshold=None, utility_margin=None)
    incubator, _ = instrument(language, monkeypatch, 0.8)
    with pytest.raises(ValueError, match="calibration"):
        incubator.turn(ScientificState("Q?"))


def test_budget_exhaustion_restores_saved_state_without_overrun(tiny, monkeypatch):
    language = tiny()
    incubator, _ = instrument(language, monkeypatch, 0.9)
    state = ScientificState("Q?", ["E"], split="train")
    cache = language.prefill(state)
    memory_cost = len(language.ids(state.canonical(), reserve=language.config.code_slots))
    activity_cost = len(language.ids(prompts.activity("Invent an object"), reserve=language.config.activity_tokens + language.config.code_slots)) + language.config.code_slots
    budget = memory_cost + activity_cost + 2
    result = incubator.turn(state, 8, cache, training=True, option_index=2, position_budget=budget)
    assert result.state is state and result.cache is cache
    assert result.event["budget_exhausted"] and not result.event["completed_incubation"]
    assert result.event["processed_positions"] <= budget


def test_interactive_rollout_passes_tools_and_closes_session():
    class Session:
        initial_observation, tools = "E", ["measure"]
        closed = False
        steps = 0
        def step(self, action):
            self.steps += 1
            self.tools = ["read"]
            return "Observed", self.steps == 2
        def score(self):
            return 70
        def close(self):
            self.closed = True
    session = Session()
    seen = []
    class Runner:
        config = SimpleNamespace(action_tokens=2, cost_scale=1000)
        language = SimpleNamespace(complete=lambda *args, **kwargs: Completion("measure", [], [], 0, 2))
        def turn(self, state, seed, training=False, option_index=None, position_budget=None):
            seen.append((state.tools, option_index))
            return SimpleNamespace(state=state, cache=object(), event={"processed_positions": 3, "incubation_positions": 1})
    result = run_episode(Runner(), SimpleNamespace(start=lambda task, seed: session),
                         {"question": "Q?", "task_id": "s", "split": "train"}, option_index=0)
    assert seen == [(["measure"], 0), (["read"], 0)]
    assert session.closed and result["native_score"] == 70
    assert result["processed_positions"] == 10 and result["complete_episode"]
    assert result["incubation_positions"] == 2
    assert "scientific_reward" not in result


def test_session_is_closed_when_split_validation_fails():
    session = SimpleNamespace(initial_observation="E", close=lambda: closed.append(True))
    closed = []
    with pytest.raises(ValueError, match="explicit compatible"):
        run_episode(object(), SimpleNamespace(start=lambda task, seed: session), {"question": "Q?"})
    assert closed == [True]

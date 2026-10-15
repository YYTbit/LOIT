import json

import pytest
import torch

from loit.calibration import gate_threshold, utility_margin
from loit.controller import admissible_actions, choose_action, ppo_loss
from loit.data import ScientificState
from loit.storage import load_language, save_checkpoint
from loit.training import Objectives, episode_return, train


def test_strict_margin_and_focus_tie(tiny):
    language = tiny()
    mask = admissible_actions(torch.tensor([[0.0, 0.5, 0.6]]), 0.5)
    assert mask.tolist() == [[True, False, True]]
    for parameter in language.model.actor.parameters():
        parameter.data.zero_()
    summary = torch.zeros(1, language.model.width + 3)
    index, _, _ = choose_action(language.model, summary)
    assert index == 0


def test_forced_action_has_its_own_likelihood(tiny):
    language = tiny()
    model = language.model
    summary = torch.zeros(1, model.width + 3)
    chosen, probability, mask = choose_action(model, summary, training=True, option_index=2)
    expected = model.actor(summary).log_softmax(-1)[0, 2]
    assert chosen == 2 and probability == pytest.approx(float(expected))
    assert mask.all()


def test_utility_reference_is_zero_and_independent_of_actor(tiny):
    model = tiny().model
    summary = torch.randn(2, model.width + 3)
    gain, cost, utility = model.predicted_utility(summary)
    assert torch.equal(utility[:, 0], torch.zeros(2))
    assert (cost >= 0).all()
    utility.sum().backward()
    assert any(p.grad is not None for p in model.utility.parameters())
    assert all(p.grad is None for p in model.actor.parameters())


def test_ppo_trains_actor_and_critic_from_complete_return(tiny):
    model = tiny().model
    model.select_trainable("controller")
    summaries = torch.randn(3, model.width + 3)
    actions = torch.tensor([0, 1, 2])
    old = model.actor(summaries).log_softmax(-1).gather(1, actions[:, None]).squeeze().detach()
    loss = ppo_loss(model, summaries, actions, old, torch.tensor([0.4, 0.8, -0.2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.critic.parameters())
    assert all(p.grad is None for p in model.utility.parameters())


def test_episode_return_requires_independent_normalized_rewards(tiny):
    config = tiny().config
    episode = {"complete_episode": True, "events": [{}], "scientific_reward": 0.6,
               "idea_reward": 0.4, "processed_positions": 900, "incubation_positions": 100}
    assert episode_return(episode, config) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        episode_return({**episode, "scientific_reward": 95}, config)
    with pytest.raises(ValueError):
        episode_return({**episode, "complete_episode": False}, config)


def test_calibration_uses_independent_development_labels(tiny):
    records = [{"split": "dev", "gate_score": 0.1, "verified": False},
               {"split": "dev", "gate_score": 0.8, "verified": True}]
    assert gate_threshold(records, 0.5) == 0.8
    with pytest.raises(ValueError):
        gate_threshold([{**records[0], "split": "test"}])
    model = tiny().model
    paired = {"split": "dev", "state_id": "s", "summary": [0.0] * (model.width + 3),
              "option_index": 1, "incubation_return": 0.6, "focused_return": 0.4, "added_cost": 0.05}
    result = utility_margin(model, [paired], 0.95)
    predicted = model.predicted_utility(torch.tensor([paired["summary"]]))[2][0, 1]
    assert result == pytest.approx(abs(float(predicted) - 0.15))


@pytest.mark.parametrize("selection", ["forced", "argmax", None])
def test_controller_rejects_unsampled_interventions(tiny, selection):
    language = tiny()
    record = {"complete_episode": True, "scientific_reward": 1, "idea_reward": 1,
              "processed_positions": 1, "incubation_positions": 1,
              "events": [{"forced_option": selection == "forced", "action_selection": selection}]}
    with pytest.raises(ValueError, match="sampled controller"):
        Objectives(language, "controller")([record])


@pytest.mark.parametrize("stage", ["focused", "initialize", "association", "return", "utility", "controller"])
def test_cpu_training_stage_saves_loadable_checkpoint(tiny, tmp_path, stage):
    language = tiny()
    state = ScientificState("Q?", ["E"], split="train").to_dict()
    proposal = {"hypothesis": "H", "competing_explanation": "C", "discriminating_test": "T",
                "prediction": "P", "competing_prediction": "R"}
    records = {
        "focused": {"state": state, "response": "Action"},
        "initialize": {"state": state},
        "return": {"state": state, "proposal": proposal, "terminal_code": 1, "verified": True},
        "utility": {"state_id": "s", "summary": [0.0] * 19, "option_index": 1,
                    "incubation_return": 0.8, "focused_return": 0.5, "added_cost": 0.1},
        "controller": {"complete_episode": True, "scientific_reward": 0.8, "idea_reward": 0.6,
                       "processed_positions": 2, "incubation_positions": 1, "events": [{"summary": [0.0] * 19, "option_index": 0,
                       "actor_log_probability": -1.6094379, "admissible": [True] * 5, "action_selection": "sampled"}]},
    }
    if stage == "association":
        memory, _ = language.memory(ScientificState.from_dict(state))
        response = language.complete("Activity", 3, 9, "association", memory, 0, 1)
        _, logits = language.score_activity(memory, 0, response.prompt_ids, response.token_ids, 1)
        records[stage] = {"state": state, "option_index": 2, "initial_code": 0, "terminal_code": 1,
                          "topic_change": language.config.topic_change, "temperature": language.config.temperature,
                          "verified_reward": 0.8,
                          "behavior_log_probability": response.log_probability + float(logits.log_softmax(-1)[1]),
                          "steps": [{"previous_code": 0, "proposed_code": 1, "next_code": 1, "reset": False,
                                     "prompt_ids": response.prompt_ids, "response_ids": response.token_ids}]}
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps({"split": "train", **records[stage]}) + "\n")
    destination = tmp_path / stage
    train(language, stage, path, destination)
    restored = load_language(language.config, destination)
    for key, parameter in language.model.state_dict().items():
        assert torch.equal(parameter.cpu(), restored.model.state_dict()[key].cpu())


def test_checkpoint_preserves_controller_and_calibration(tiny, tmp_path):
    language = tiny()
    save_checkpoint(language, tmp_path)
    restored = load_language(language.config, tmp_path)
    assert restored.config.gate_threshold == language.config.gate_threshold
    tokens = language.tensor([3, 4])
    assert torch.equal(language.model(tokens).logits, restored.model(tokens).logits)

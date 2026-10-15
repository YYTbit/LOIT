import pytest
import torch
import json

from loit.data import ScientificState


@pytest.mark.parametrize("family", ["qwen3", "llama", "qwen2"])
def test_shared_backbone_conditioning_and_zero_coupling(tiny, family):
    language = tiny(family)
    model = language.model
    tokens = language.tensor([3, 4, 5])
    memory = torch.randn(1, 2, model.width)
    left = model(tokens, "association", memory, 0, 0).logits
    right = model(tokens, "association", memory + 2, 0, 0).logits
    assert torch.equal(left, right)
    conditioned = model(tokens, "association", memory, 0, 1).logits
    assert not torch.allclose(left, conditioned)
    focused = model(tokens).logits
    returned = model(tokens, "return", code=0).logits
    assert focused.shape == returned.shape == (1, 3, model.backbone.config.vocab_size)
    assert not torch.allclose(focused, returned)
    assert model._conditioning is None


def test_frozen_backbone_transmits_association_gradients(tiny):
    language = tiny()
    model = language.model
    model.select_trainable("association")
    memory, _ = language.memory(ScientificState("Question?", ["Evidence"]))
    score, transition = language.score_activity(memory, 0, [4, 5], [6, 7], 1)
    loss = -score - transition.log_softmax(-1)[1]
    loss.backward()
    assert model.codebook.grad is None
    assert all(p.grad is None for p in model.memory.parameters())
    for module in [model.association_attention, model.transition]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is None for p in model.return_attention.parameters())


def test_response_likelihood_matches_cached_generation(tiny):
    language = tiny(temperature=0.7)
    memory = torch.randn(1, 2, language.model.width)
    completion = language.complete("Activity", 5, 11, "association", memory, 1, 1)
    score, _ = language.score_activity(memory, 1, completion.prompt_ids, completion.token_ids, 1)
    assert float(score) == pytest.approx(completion.log_probability, abs=1e-5)


def test_saved_focused_cache_survives_return_generation(tiny):
    language = tiny()
    state = ScientificState("Question?", ["Evidence"])
    cache = language.prefill(state)
    length = cache.values.get_seq_length()
    logits = cache.logits.clone()
    language.complete("Return", 3, branch="return", code=1, cache=cache)
    assert cache.values.get_seq_length() == length
    assert torch.equal(cache.logits, logits)


def test_return_supervision_updates_return_branch(tiny):
    language = tiny()
    language.model.select_trainable("return")
    language.return_loss(ScientificState("Q?"), 1, '{"hypothesis":"test"}').backward()
    assert language.model.codebook.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in language.model.return_attention.parameters())
    assert any(p.grad is not None for p in language.model.return_gates.parameters())
    assert all(p.grad is None for p in language.model.backbone.parameters())


def test_initialization_learns_memory_and_code_representations(tiny):
    language = tiny()
    model = language.model
    model.select_trainable("initialize")
    memory, _ = language.memory(ScientificState("Q?", ["E"]))
    language.continuation_loss("Reconstruct", "Q and E", "association", memory, 0, 1).backward()
    assert model.codebook.grad is not None and model.codebook.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.memory.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())


def test_conditioning_is_cleared_after_exception(tiny):
    model = tiny().model
    with pytest.raises(RuntimeError):
        with model.condition("return", code=0):
            raise RuntimeError("interrupted forward")
    assert model._conditioning is None


def test_grammar_receives_one_growing_prefix(tiny, monkeypatch):
    language = tiny()
    seen = []
    token = language.tokenizer.encode("a", add_special_tokens=False)[0]
    def grammar(batch, prefix):
        seen.append(prefix.tolist())
        return [token]
    monkeypatch.setattr(language, "grammar", lambda schema: grammar)
    result = language.complete("P", 3, schema={"type": "string"})
    assert result.text == "aaa"
    assert seen == [result.prompt_ids + result.token_ids[:step] for step in range(3)]


def test_json_schema_decoder_produces_a_valid_structured_response(tiny):
    language = tiny(grammar=True)
    schema = {"type": "object", "properties": {"x": {"type": "integer", "enum": [0]}},
              "required": ["x"], "additionalProperties": False}
    result = language.complete("Q", 128, seed=4, schema=schema)
    assert json.loads(result.text) == {"x": 0}


def test_context_overflow_does_not_discard_question(tiny):
    language = tiny(context_tokens=512)
    state = ScientificState("Preserve this question", ["x" * 1000])
    with pytest.raises(ValueError, match="retaining the question"):
        language.prefill(state)


def test_generation_respects_decoder_position_budget(tiny):
    from loit.language import ComputationBudgetExceeded
    language = tiny()
    result = language.complete("P", 128, position_budget=4)
    assert result.processed_tokens <= 4
    assert len(result.token_ids) <= 3
    with pytest.raises(ComputationBudgetExceeded):
        language.complete("A long prompt", 3, position_budget=1)

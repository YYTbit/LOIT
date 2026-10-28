import json
import random

import numpy as np
import pytest

from loit.config import Config
from loit.data import ActivityPool, ScientificState, prepare_activities
from loit.theory import contraction_bound, coupling_derivative, escape_certificate, kernel, useful_probability
from loit.training import RecordDataset


def test_residual_memory_excludes_hypotheses_and_reference_answers():
    state = ScientificState.from_dict({"question": "Q", "observations": ["E"], "proposals": ["H"],
                                       "reference_answer": "SECRET", "verifier": "PRIVATE"})
    assert json.loads(state.canonical()) == {"question": "Q", "observations": ["E"]}
    assert "SECRET" not in json.dumps(state.to_dict())
    assert state.split == "unspecified"


def test_activity_filtering_deduplication_and_source_answer_exclusion():
    source = [
        {"category": "creative_writing", "instruction": "Invent an object", "response": "SECRET"},
        {"category": "creative_writing", "instruction": "  invent AN object  "},
        {"category": "summarization", "instruction": "Summarize the following text"},
        {"category": "summarization", "instruction": "Summarize the following text", "context": "Visible text"},
        {"category": "open_qa", "instruction": "DiscoveryWorld question"},
    ]
    records, counts = prepare_activities(source, 2, ["DiscoveryWorld"])
    assert counts == {"generation": 1, "summarization": 1}
    assert "SECRET" not in json.dumps(records)
    assert records == prepare_activities(source, 2, ["DiscoveryWorld"])[0]
    assert any("Visible text" in record["instruction"] for record in records)


def test_activity_sampler_depends_on_category_and_seed():
    pool = ActivityPool([{"activity_id": "a", "category": "x", "instruction": "One"},
                         {"activity_id": "b", "category": "x", "instruction": "Two"}], ["x"])
    assert pool.sample("x", random.Random(8)) == pool.sample("x", random.Random(8))
    with pytest.raises(ValueError, match="Each configured"):
        ActivityPool([], ["x"])


@pytest.mark.parametrize("split", [None, "dev", "test"])
def test_training_rejects_implicit_or_reserved_splits(tmp_path, split):
    record = {"state": {"question": "Q"}, "response": "A"}
    if split is not None:
        record["split"] = split
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="explicitly designated"):
        RecordDataset(path)[0]


def test_reset_kernel_and_finite_horizon_probability():
    proposal = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.4, 0.6])
    transition = kernel(proposal, initial, 0.24)
    assert np.allclose(transition.sum(1), 1)
    assert np.allclose(kernel(proposal, initial, 1), np.tile(initial, (2, 1)))
    expected = initial.copy()
    for _ in range(5):
        expected = expected @ transition
    quality = np.array([0.2, 0.8])
    assert useful_probability(initial, transition, quality, 5) == pytest.approx(expected @ quality)
    assert useful_probability(initial, transition, quality, 0) == pytest.approx(initial @ quality)


def test_coupling_derivative_matches_finite_difference():
    initial, quality = [0.3, 0.7], [0.8, 0.1]
    transition = np.array([[0.7, 0.3], [0.4, 0.6]])
    derivative = np.array([[-0.1, 0.1], [0.2, -0.2]])
    step = 1e-6
    numerical = (useful_probability(initial, transition + step * derivative, quality, 4)
                 - useful_probability(initial, transition - step * derivative, quality, 4)) / (2 * step)
    assert coupling_derivative(initial, transition, derivative, quality, 4) == pytest.approx(numerical, abs=1e-9)


def test_reset_contraction_and_confidence_radius():
    transition = kernel([[0.95, 0.05], [0.05, 0.95]], [0.5, 0.5], 0.24)
    spread = np.array([1.0, 0.0]) @ np.linalg.matrix_power(transition, 6) - np.array([0.0, 1.0]) @ np.linalg.matrix_power(transition, 6)
    assert np.abs(spread).sum() / 2 <= contraction_bound(0.24, 6)
    small = escape_certificate([0.5, 0.5], transition, [0.2, 0.8], 4, 100, 100)
    large = escape_certificate([0.5, 0.5], transition, [0.2, 0.8], 4, 100000, 100000)
    assert large["radius"] < small["radius"]
    assert 0 <= large["lower"] <= large["estimate"] <= large["upper"] <= 1


@pytest.mark.parametrize("field", ["transition_samples", "return_samples"])
@pytest.mark.parametrize("count", [0, -1, True, 1.5, float("nan"), float("inf")])
def test_certificate_rejects_invalid_sample_counts(field, count):
    counts = {"transition_samples": 100, "return_samples": 100}
    counts[field] = count
    with pytest.raises(ValueError, match="positive finite integer"):
        escape_certificate([0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]], [0.2, 0.8], 3, **counts)


def test_certificate_accepts_finite_integer_valued_counts():
    arguments = ([0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]], [0.2, 0.8], 3)
    assert escape_certificate(*arguments, 100.0, 100.0) == escape_certificate(*arguments, 100, 100)


@pytest.mark.parametrize("values", [{"topic_change": -0.1}, {"couplings": [1.2]}, {"gate_threshold": 1.1}, {"gate_weights": [1, 1, 1, 1]}])
def test_invalid_execution_settings_are_rejected(values):
    with pytest.raises(ValueError):
        Config(**values)

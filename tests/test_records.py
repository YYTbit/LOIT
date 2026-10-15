from copy import deepcopy

import pytest

from loit.records import attach_labels, identify_records


def test_verification_adds_rewards_without_changing_visible_state():
    records = identify_records([{"state": {"question": "Q?", "split": "train"}, "steps": [{"next_code": 2}]}])
    before = deepcopy(records)
    labels = [{"record_id": records[0]["record_id"], "split": "train", "verified_reward": 0.8,
               "reference_answer": "SECRET", "state": {"question": "changed"}}]
    result = attach_labels(records, labels, "association")
    assert records == before
    assert result[0] == {**records[0], "verified_reward": 0.8}


@pytest.mark.parametrize("split", ["test", None])
def test_verification_rejects_split_mismatch(split):
    record = identify_records([{"split": "train", "events": []}])[0]
    label = {"record_id": record["record_id"], "split": split, "scientific_reward": 0.5, "idea_reward": 0.8}
    with pytest.raises(ValueError, match="matching explicit splits"):
        attach_labels([record], [label], "controller")


def test_verification_rejects_unused_or_duplicate_labels():
    records = identify_records([{"split": "train"}])
    label = {"record_id": records[0]["record_id"], "split": "train", "verified_reward": 1}
    with pytest.raises(ValueError, match="duplicate"):
        attach_labels(records, [label, label], "association")
    with pytest.raises(ValueError, match="uncollected"):
        attach_labels(records, [label, {**label, "record_id": "other"}], "association")

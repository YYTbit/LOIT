from copy import deepcopy
import hashlib
import math

from .data import stable_text


def identify_records(records):
    identified = []
    seen = set()
    for record in records:
        value = dict(record)
        identifier = value.get("record_id") or hashlib.sha256(stable_text(value).encode()).hexdigest()
        if identifier in seen:
            raise ValueError("Collected records require distinct identifiers")
        seen.add(identifier)
        value["record_id"] = identifier
        identified.append(value)
    return identified


def attach_labels(records, labels, stage):
    fields = {"association": ["verified_reward"], "controller": ["scientific_reward", "idea_reward"]}[stage]
    indexed = {}
    for label in labels:
        identifier = label["record_id"]
        if identifier in indexed:
            raise ValueError("Independent labels contain a duplicate record identifier")
        indexed[identifier] = label
    output = []
    for record in records:
        identifier = record["record_id"]
        if identifier not in indexed:
            raise ValueError("Each collected record requires independent verification")
        label = indexed.pop(identifier)
        split = record.get("split", record.get("state", {}).get("split"))
        if not split or label.get("split") != split:
            raise ValueError("Collected records and independent labels require matching explicit splits")
        value = deepcopy(record)
        # Verification adds reward fields without changing the sampled trajectory or visible state.
        for field in fields:
            reward = float(label[field])
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise ValueError("Independently verified rewards require [0, 1]")
            value[field] = reward
        output.append(value)
    if indexed:
        raise ValueError("Independent labels refer to uncollected records")
    return output

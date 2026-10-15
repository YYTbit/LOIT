import math
import numpy as np
import torch


def gate_threshold(records, acceptance_floor=0.1):
    if not records or not 0 < acceptance_floor <= 1:
        raise ValueError("Gate calibration requires records and a positive acceptance floor")
    scores, labels = [], []
    for record in records:
        if record.get("split") not in ["dev", "validation"]:
            raise ValueError("Calibration requires development records")
        score, label = float(record["gate_score"]), record["verified"]
        if not 0 <= score <= 1 or label not in [0, 1, False, True]:
            raise ValueError("Gate scores and independent binary labels are invalid")
        scores.append(score)
        labels.append(int(label))
    choices = []
    for threshold in sorted(set([0.0] + scores)):
        selected = [label for score, label in zip(scores, labels) if score >= threshold]
        if len(selected) / len(scores) >= acceptance_floor:
            choices.append((sum(selected) / len(selected), len(selected), threshold))
    return max(choices)[2]


@torch.no_grad()
def utility_margin(model, records, quantile=0.95):
    if not records or not 0 < quantile <= 1:
        raise ValueError("Utility calibration requires paired records and a valid coverage quantile")
    errors = []
    for record in records:
        if record.get("split") not in ["dev", "validation"] or not record.get("state_id"):
            raise ValueError("Utility calibration requires paired development states")
        summary = torch.tensor([record["summary"]], dtype=torch.float32, device=model.device)
        if not torch.isfinite(summary).all() or summary.shape != (1, model.width + 3):
            raise ValueError("The paired controller summary is invalid")
        if not 0 < record["option_index"] < len(model.config.options) or not math.isfinite(record["added_cost"]) or record["added_cost"] < 0:
            raise ValueError("Paired development records require an incubation option and nonnegative added cost")
        _, _, predicted = model.predicted_utility(summary)
        actual = float(record["incubation_return"]) - float(record["focused_return"]) - model.config.cost_weight * float(record["added_cost"])
        if not math.isfinite(actual):
            raise ValueError("The paired development utility is nonfinite")
        errors.append(abs(float(predicted[0, record["option_index"]]) - actual))
    # This is an empirical validation margin, distinct from a population coverage guarantee.
    return float(np.quantile(errors, quantile, method="higher"))

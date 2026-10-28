import math
import numpy as np


def distribution(value, size=None):
    value = np.asarray(value, dtype=float)
    if value.ndim != 1 or (size is not None and len(value) != size) or not np.all(np.isfinite(value)) or np.any(value < 0) or not np.isclose(value.sum(), 1):
        raise ValueError("A finite probability vector requires nonnegative entries and unit sum")
    return value


def kernel(proposal, initial, epsilon):
    proposal = np.asarray(proposal, dtype=float)
    if proposal.ndim != 2 or proposal.shape[0] != proposal.shape[1]:
        raise ValueError("The proposal matrix requires a square shape")
    initial = distribution(initial, len(proposal))
    for row in proposal:
        distribution(row, len(proposal))
    if not 0 <= epsilon <= 1:
        raise ValueError("The topic-change probability requires [0, 1]")
    return (1 - epsilon) * proposal + epsilon * np.ones((len(initial), 1)) * initial


def useful_probability(initial, transition, quality, duration):
    transition = np.asarray(transition, dtype=float)
    initial = distribution(initial, len(transition))
    quality = np.asarray(quality, dtype=float)
    if duration < 0 or int(duration) != duration or quality.shape != initial.shape or np.any((quality < 0) | (quality > 1)) or not np.all(np.isfinite(quality)):
        raise ValueError("Duration or terminal-code quality is invalid")
    for row in transition:
        distribution(row, len(initial))
    return float(initial @ np.linalg.matrix_power(transition, duration) @ quality)


def coupling_derivative(initial, transition, derivative, quality, duration):
    initial = distribution(initial)
    transition, derivative, quality = map(lambda value: np.asarray(value, dtype=float), [transition, derivative, quality])
    if transition.shape != derivative.shape or transition.shape != (len(initial), len(initial)) or quality.shape != initial.shape:
        raise ValueError("The derivative and transition dimensions differ")
    if duration < 0 or int(duration) != duration:
        raise ValueError("Duration requires a nonnegative integer")
    # Differentiating each factor retains every memory-access contribution.
    return float(sum(initial @ np.linalg.matrix_power(transition, step) @ derivative @ np.linalg.matrix_power(transition, duration - 1 - step) @ quality for step in range(duration)))


def contraction_bound(epsilon, duration):
    if not 0 < epsilon <= 1 or duration < 0 or int(duration) != duration:
        raise ValueError("The contraction certificate requires positive topic changes and an integer horizon")
    return (1 - epsilon) ** duration


def escape_certificate(initial, transition, quality, duration, transition_samples, return_samples, delta=0.05):
    size = len(initial)
    counts = (transition_samples, return_samples)
    if any(isinstance(count, (bool, np.bool_)) or not math.isfinite(count)
           or count < 1 or int(count) != count for count in counts):
        raise ValueError("The certificate requires positive finite integer sample counts")
    if not 0 < delta < 1:
        raise ValueError("The certificate requires confidence in (0, 1)")
    estimate = useful_probability(initial, transition, quality, duration)
    row_error = min(2.0, size * math.sqrt(math.log(4 * size * size / delta) / (2 * transition_samples)))
    quality_error = min(1.0, math.sqrt(math.log(4 * size / delta) / (2 * return_samples)))
    radius = min(1.0, duration * row_error / 2 + quality_error)
    return {"estimate": estimate, "radius": radius, "lower": max(0.0, estimate - radius), "upper": min(1.0, estimate + radius)}

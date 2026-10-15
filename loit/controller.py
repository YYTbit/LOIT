import math

import torch
from torch.nn import functional as F


def admissible_actions(utilities, margin):
    if margin is None or not math.isfinite(margin) or margin < 0 or not torch.isfinite(utilities).all():
        raise ValueError("Deployment requires a calibrated margin and finite utility estimates")
    mask = utilities > margin
    mask[..., 0] = True
    return mask


def action_distribution(logits, mask=None):
    if mask is not None:
        if mask.shape != logits.shape or not mask.any(dim=-1).all():
            raise ValueError("Each policy row requires an admissible action")
        logits = logits.masked_fill(~mask, -torch.inf)
    return torch.distributions.Categorical(logits=logits.float())


def choose_action(model, summary, training=False, seed=0, option_index=None):
    logits = model.actor(summary)
    _, _, utility = model.predicted_utility(summary)
    mask = torch.ones_like(logits, dtype=torch.bool) if training else admissible_actions(utility, model.config.utility_margin)
    policy = action_distribution(logits, mask)
    if option_index is not None:
        if not 0 <= option_index < logits.shape[-1] or not mask[0, option_index]:
            raise ValueError("The requested incubation option is invalid or inadmissible")
        index = option_index
    elif training:
        generator = torch.Generator(device=logits.device).manual_seed(seed)
        index = int(torch.multinomial(policy.probs[0], 1, generator=generator))
    else:
        # argmax chooses index zero when focused continuation ties for first place.
        index = int(policy.probs[0].argmax())
    return index, float(policy.log_prob(torch.tensor([index], device=logits.device))[0]), mask[0]


def paired_loss(model, summaries, actions, gains, costs):
    predicted_gain, predicted_cost, _ = model.predicted_utility(summaries)
    index = actions[:, None]
    gain_loss = F.mse_loss(predicted_gain.gather(1, index).squeeze(1), gains)
    cost_loss = F.mse_loss(predicted_cost.gather(1, index).squeeze(1), costs)
    return gain_loss + model.config.cost_regression_weight * cost_loss


def ppo_loss(model, summaries, actions, old_log_probabilities, returns, masks=None):
    policy = action_distribution(model.actor(summaries), masks)
    values = model.critic(summaries).squeeze(-1)
    advantages = returns - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    ratios = (policy.log_prob(actions) - old_log_probabilities).exp()
    clipped = ratios.clamp(1 - model.config.ppo_clip, 1 + model.config.ppo_clip)
    actor = -torch.minimum(ratios * advantages, clipped * advantages).mean()
    critic = F.mse_loss(values, returns)
    return actor + critic - model.config.entropy_weight * policy.entropy().mean()

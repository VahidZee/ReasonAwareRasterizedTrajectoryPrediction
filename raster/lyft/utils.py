import torch
import numpy as np


def filter_batch(batch: dict, filter_static_history: float):
    tar_idxs = (batch['target_availabilities'].sum(axis=1) == batch['target_availabilities'].shape[1])

    # filter scenes with low history availabilities
    his_idxs = (batch['history_availabilities'].sum(axis=1) == batch['history_availabilities'].shape[1])

    # filter scenes with static history
    if batch['target_availabilities'].shape[1] > 1 and filter_static_history:
        diff = batch['history_positions'][:, -1] - batch['history_positions'][:, 0]
        stat_idxs = (diff.norm(p=2, dim=1) > filter_static_history)
    else:
        stat_idxs = torch.ones(tar_idxs.shape[0], dtype=torch.bool)

    # applying filters
    idxs = (tar_idxs * his_idxs * stat_idxs).data
    for key in batch:
        batch[key] = batch[key][idxs]
    return idxs.sum().data


def find_batch_extremes(batch, batch_loss, outputs, k=5, require_grad=False, to_tensor=False):
    assert batch['track_id'].shape[0] == batch_loss.shape[0], 'batch and loss shape miss-match'
    assert batch['track_id'].shape[0] == outputs.shape[0], 'batch and outputs shape miss-match'
    if to_tensor:
        batch = {key: torch.from_numpy(value).to(batch_loss.device) for key, value in batch.items()}
    with torch.set_grad_enabled(require_grad):
        worst_loss, worst_idx = torch.topk(batch_loss, min(k, batch_loss.shape[0]))
        best_loss, best_idx = torch.topk(batch_loss, min(k, batch_loss.shape[0]), largest=False)
        best_batch = {key: value[best_idx] for key, value in batch.items()}
        best_batch['loss'] = best_loss
        best_batch['prediction'] = outputs[best_idx]
        best_batch['outputs'] = outputs[best_idx]
        worst_batch = {key: value[worst_idx] for key, value in batch.items()}
        worst_batch['loss'] = worst_loss
        worst_batch['prediction'] = outputs[worst_idx]
        worst_batch['outputs'] = outputs[worst_idx]
    return best_batch, worst_batch


# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
def neg_multi_log_likelihood(
        gt: torch.Tensor, pred: torch.Tensor, confidences: torch.Tensor, avails: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    batch_size, num_modes, future_len, num_coords = pred.shape

    # convert to (batch_size, num_modes, future_len, num_coords)
    if len(gt.shape) != len(pred.shape):
        gt = torch.unsqueeze(gt, 1)  # add modes
    if avails is not None:
        avails = avails[:, None, :, None]  # add modes and cords
        # error (batch_size, num_modes, future_len)
        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability
    else:
        error = torch.sum((gt - pred) ** 2, dim=-1)  # reduce coords and use availability
    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    return error.reshape(-1)

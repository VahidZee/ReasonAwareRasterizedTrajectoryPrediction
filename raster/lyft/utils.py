import torch


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

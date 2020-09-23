import torch
import numpy as np
import matplotlib.pyplot as plt

from l5kit.geometry import transform_points
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory, PREDICTED_POINTS_COLOR
from captum.attr import Saliency
from captum.attr import visualization as viz

from typing import Union, Optional, Any, Tuple


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


def trajectory_stat(
        history_positions: np.array,
        target_positions: np.array,
        centroid: np.array,
        world_to_image: np.array,
        threshold: Optional[float] = 3.,
        prefix: Optional[Any] = 'data/',
        matrix=False
) -> Union[tuple, str]:
    target_pixels = transform_points(target_positions + centroid, world_to_image)
    target_pixels -= target_pixels[0]
    target_change = target_pixels[np.argmax(np.abs(target_pixels[:, 1])), 1]
    if np.abs(target_change) > threshold:
        target = 'D' if target_change < 0. else 'U'
    else:
        target = 'S'

    history_pixels = transform_points(history_positions + centroid, world_to_image)
    history_pixels -= history_pixels[0]
    history_change = history_pixels[np.argmax(np.abs(history_pixels[:, 1])), 1]
    if np.abs(history_change) > threshold:
        history = 'U' if history_change < 0. else 'D'
    else:
        history = 'S'
    if matrix:
        conv = lambda x: 1 if x == 'S' else 0 if x == 'U' else 2
        return conv(history), conv(target)
    return f'{prefix}{history}{target}'


def batch_stats(batch: dict, targets: Optional[torch.Tensor] = None, threshold: Optional[float] = 3.,
                prefix: Optional[Any] = 'data/', matrix=False) -> Union[dict, np.array]:
    """
    Get a confusion statistic path directions in the batch.
    :param batch: batch
    :param targets: predicted targets
    :param threshold: manhattan distance threshold in pixels
    :param prefix:
    :param matrix: if set to true a confusion matrix of (U,S,D) will be returned
    :return: depending on :matrix: will either be a confusion dictionary or matrix
    """
    targets = targets if targets is not None else batch["target_positions"]
    batch_size = targets.shape[0]
    if matrix:
        result = np.zeros((3, 3), dtype=np.float32)
    else:
        result = {f'{prefix}{i}{j}': torch.zeros(1, dtype=torch.float32) for i in 'SUD' for j in 'SUD'}
    for hist, future, cent, wti in zip(batch['history_positions'], targets, batch['centroid'],
                                       batch['world_to_image']):
        temp = trajectory_stat(hist.cpu().data.numpy(), future.cpu().data.numpy(), cent.cpu().data.numpy(),
                               wti.cpu().data.numpy(), threshold, prefix, matrix=matrix)
        if matrix:
            result[temp[0]][temp[1]] += 1
        else:
            result[temp] += 1
    if matrix:
        result /= batch_size
    else:
        for key in result:
            result[key] /= batch_size
    return result


def _set_image_type(image: np.array):
    """
    Utility function to force the type of image pixels,
        A mysterious bug caused the torch.Tensor.numpy() to miss produce the resulting array's type
    :param image:
    :return:
    """
    image_np = image
    float_image = np.zeros(image_np.shape)
    for i, channel in enumerate(image_np):
        for j, row in enumerate(channel):
            for z, pixel in enumerate(row):
                float_image[i][j][z] = float(pixel)
    return float_image


def draw_single_image(
        rasterizer,
        image: np.array,
        centroid: np.array,
        world_to_image: np.array,
        target_positions: np.array,
        target_yaws: np.array,
        predicted_positions: Optional[np.array] = None,
        predicted_yaws: Optional[np.array] = None,
        target_color: Optional[tuple] = TARGET_POINTS_COLOR,
        predicted_color: Optional[tuple] = PREDICTED_POINTS_COLOR,
) -> torch.Tensor:
    """
    Produce a single RGB representation of the rasterized input image and its corresponding position prediction
    :param rasterizer:
    :param image:
    :param centroid:
    :param world_to_image:
    :param target_positions:
    :param target_yaws:
    :param predicted_positions:
    :param predicted_yaws:
    :param target_color:
    :param predicted_color:
    :return:
    """
    predicted_yaws = predicted_yaws if predicted_yaws is not None else target_yaws
    im = _set_image_type(rasterizer.to_rgb(image.cpu().data.numpy().transpose(1, 2, 0)))  # Todo enhance
    draw_trajectory(im, transform_points(
        target_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(), world_to_image.cpu().data.numpy()),
                    target_yaws.cpu().data.numpy(), target_color)
    if predicted_positions is not None:
        draw_trajectory(im, transform_points(
            predicted_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(),
            world_to_image.cpu().data.numpy()), predicted_yaws.cpu().data.numpy(), predicted_color)
    return np.uint8(im).transpose(2, 0, 1)


def draw_batch(rasterizer, batch: dict, outputs: Optional[torch.Tensor] = None,
               predicted_yaws: Optional[torch.Tensor] = None,
               target_color: Optional[tuple] = TARGET_POINTS_COLOR,
               predicted_color: Optional[tuple] = PREDICTED_POINTS_COLOR) -> np.array:
    """
    Creates a grid numpy array of RGB representation of the rasterized inputs and the corresponding
    predicted trajectory
    :param rasterizer: rasterizer used to create the batch
    :param batch: batch dictionary
    :param outputs:
    :param predicted_yaws:
    :param target_color: tuple of (r,g,b)
    :param predicted_color: tuple of (r,g,b)
    :return: a grid of rgb representation of the batch
    """
    return np.array([draw_single_image(
        rasterizer, image, batch['centroid'][i], batch['world_to_image'][i], batch['target_positions'][i],
        batch['target_yaws'][i], None if outputs is None else outputs[i],
        None if predicted_yaws is None else predicted_yaws[i], target_color, predicted_color) for i, image in
        enumerate(batch['image'])],
        dtype=np.uint8)


def saliency_map(
        batch: dict,
        saliency: Saliency,
        sign: str = 'all',
        method: str = 'blended_heat_map',
        use_pyplot: bool = False,
) -> Tuple[Any, torch.Tensor]:
    """
    :param batch: batch to visualise
    :param saliency: Saliency object initialised for trainer_module
    :param sign: sign of gradient attributes to visualise
    :param method: method of visualization to be used
    :param use_pyplot: whether to use pyplot
    :return: pair of figure and corresponding gradients tensor
    """

    batch['image'].requires_grad = True
    grads = saliency.attribute(batch['image'], abs=False, additional_forward_args=(
        batch['target_positions'], None if 'target_availabilities' not in batch else batch['target_availabilities'],
        False))
    batch['image'].requires_grad = False
    gradsm = grads.squeeze().cpu().detach().numpy()
    if len(gradsm.shape) == 3:
        gradsm = gradsm.reshape(1, *gradsm.shape)
    gradsm = np.transpose(gradsm, (0, 2, 3, 1))
    im = batch['image'].detach().cpu().numpy().transpose(0, 2, 3, 1)
    fig, axis = plt.subplots(1, im.shape[0], dpi=200, figsize=(6, 6))
    fig.set_tight_layout(True)
    for b in range(im.shape[0]):
        viz.visualize_image_attr(
            gradsm[b, :, :, :], im[b, :, :, :], method=method, sign=sign,
            use_pyplot=use_pyplot, plt_fig_axis=(fig, axis if im.shape[0] == 1 else axis[b]), )
        if 'loss' in batch:
            (axis if im.shape[0] == 1 else axis[b]).set_title(f'loss: {batch["loss"][b].detach().cpu()}')
    return fig, grads

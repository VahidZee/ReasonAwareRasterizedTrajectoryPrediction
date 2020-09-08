import torch
import numpy as np
from l5kit.geometry import transform_points, transform_point
from l5kit.visualization.utils import ARROW_LENGTH_IN_PIXELS, ARROW_THICKNESS_IN_PIXELS, ARROW_TIP_LENGTH_IN_PIXELS
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory, PREDICTED_POINTS_COLOR
from typing import Optional


def find_batch_extremes(batch, batch_loss, outputs, best_k=5, worst_k=5, require_grad=False):
    assert batch['track_id'].shape[0] == batch_loss.shape[0], 'batch and loss shape miss-match'
    assert batch['track_id'].shape[0] == outputs.shape[0], 'batch and outputs shape miss-match'
    with torch.set_grad_enabled(require_grad):
        batch_loss = batch_loss.mean(1).mean(1)
        worst_loss, worst_idx = torch.topk(batch_loss, min(worst_k, len(batch_loss)))
        best_loss, best_idx = torch.topk(batch_loss, min(best_k, len(batch_loss)), largest=False)
        best_batch = {key: value[best_idx] for key, value in batch.items()}
        best_batch['loss'] = best_loss
        best_batch['prediction'] = outputs[best_idx]
        worst_batch = {key: value[worst_idx] for key, value in batch.items()}
        worst_batch['loss'] = worst_loss
        worst_batch['prediction'] = outputs[worst_idx]
    return best_batch, worst_batch


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
    predicted_yaws = predicted_yaws or target_yaws
    im = _set_image_type(rasterizer.to_rgb(image.cpu().data.numpy().transpose(1, 2, 0)))  # Todo enhance
    draw_trajectory(im, transform_points(target_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(),
                                         world_to_image.cpu().data.numpy()), target_yaws.cpu().data.numpy(),
                    target_color)
    if predicted_positions is not None:
        draw_trajectory(im, transform_points(predicted_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(),
                                             world_to_image.cpu().data.numpy()), predicted_yaws.cpu().data.numpy(),
                        predicted_color)
    return np.uint8(im[::-1])


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

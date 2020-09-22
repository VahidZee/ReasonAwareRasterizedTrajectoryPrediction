import numpy as np
import torch
from typing import Optional


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
    im = _set_image_type(rasterizer.to_rgb(image.transpose(1, 2, 0)))  # Todo enhance
    draw_trajectory(im, transform_points(
        target_positions + centroid[:2], world_to_image), target_yaws, target_color)
    if predicted_positions is not None:
        draw_trajectory(im, transform_points(
            predicted_positions + centroid[:2],
            world_to_image), predicted_yaws, predicted_color)
    return np.uint8(im).transpose(2, 0, 1)


def draw(rasterizer, traj: dict):
    im = draw_single_image(
        rasterizer, traj['image'], traj['centroid'], traj['world_to_image'],
        traj['target_positions'], traj['target_yaws']
    )
    plt.imshow(im.transpose(1, 2, 0))

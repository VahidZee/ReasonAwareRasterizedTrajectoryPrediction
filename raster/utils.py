import torch
import numpy as np
from l5kit.geometry import transform_points, transform_point
from l5kit.visualization.utils import ARROW_LENGTH_IN_PIXELS, ARROW_THICKNESS_IN_PIXELS, ARROW_TIP_LENGTH_IN_PIXELS
from l5kit.visualization import TARGET_POINTS_COLOR, draw_trajectory, PREDICTED_POINTS_COLOR
from typing import Optional, Any

from captum.attr import Saliency
from captum.attr import visualization as viz


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
        prefix: Optional[Any] = 'data/'
):
    target_pixels = transform_points(target_positions + centroid, world_to_image)
    target_pixels -= target_pixels[0]
    target_change = target_pixels[np.argmax(np.abs(target_pixels[:, 1])), 1]
    if np.abs(target_change) > threshold:
        target = 'U' if target_change < 0. else 'D'
    else:
        target = 'S'

    history_pixels = transform_points(history_positions + centroid, world_to_image)
    history_pixels -= history_pixels[0]
    history_change = history_pixels[np.argmax(np.abs(history_pixels[:, 1])), 1]
    if np.abs(history_change) > threshold:
        history = 'D' if history_change < 0. else 'U'
    else:
        history = 'S'

    return f'{prefix}{history}{target}'


def batch_stats(batch: dict, threshold: Optional[float] = 3., prefix: Optional[Any] = 'data/'):
    """
    Get a confusion statistic path directions in the batch.
    :param batch: batch
    :param threshold: manhattan distance threshold in pixels
    :param prefix:
    :return: confusion dictionary
    """
    targets = batch["target_positions"]
    batch_size = targets.shape[0]
    result = {f'{prefix}{i}{j}': torch.zeros(1, dtype=torch.float32) for i in 'SUD' for j in 'SUD'}
    for hist, future, cent, wti in zip(batch['history_positions'], batch["target_positions"], batch['centroid'],
                                       batch['world_to_image']):
        result[trajectory_stat(hist.cpu().data.numpy(), future.cpu().data.numpy(), cent.cpu().data.numpy(),
                               wti.cpu().data.numpy(), threshold, prefix)] += 1

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
    predicted_yaws = predicted_yaws or target_yaws
    im = _set_image_type(rasterizer.to_rgb(image.cpu().data.numpy().transpose(1, 2, 0)))  # Todo enhance
    draw_trajectory(im, transform_points(
        target_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(), world_to_image.cpu().data.numpy()),
                    target_yaws.cpu().data.numpy(), target_color)
    if predicted_positions is not None:
        draw_trajectory(im, transform_points(
            predicted_positions.cpu().data.numpy() + centroid[:2].cpu().data.numpy(),
            world_to_image.cpu().data.numpy()), predicted_yaws.cpu().data.numpy(), predicted_color)
    return np.uint8(im[::-1]).transpose(2, 0, 1)


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

def draw_sailiency_map(batch,saliency):
    dictionary = {}
    # print(batch['target_positions'].shape)
    # print(batch['image'].shape)
    print("here")
    batch_num = batch['target_positions'].shape[0]
    future_len = batch['target_positions'].shape[1]
    raster_len = batch['image'].shape[1]
    im = batch['image'].detach().numpy().transpose(0, 2, 3, 1)
    half_way = int((raster_len-3)/2)
    # print(half_way)
    # for j in range(len(batch['target_positions'][0])):
    for j in range(future_len):
        print(j)
        grads = saliency.attribute(batch['image'], target=2 * j)
        gradsp = saliency.attribute(batch['image'], target=2 * j + 1)
        # if j == 0  or j == future_len-1:
        #     gradsm = (grads + gradsp) / 2
        #     gradsm = np.transpose(gradsm.squeeze().cpu().detach().numpy(), (0,2, 3, 1))
        #     for b in range(batch_num) :
        #         print(b)
        #         fig, axis = viz.visualize_image_attr(gradsm[b,:,:,0:1], im[b,:,:,0:1],
        #                                                  method="blended_heat_map", sign="absolute_value",
        #                                                  show_colorbar=True, title="Overlayed Gradient Magnitudes")
        #         dictionary['plot_batchNum'+ str(b) + '_Future' + str(j * 2) + '_Raster' + str(0) ] = fig
        #         fig, axis = viz.visualize_image_attr(gradsm[b, :, :, half_way:half_way+1], im[b, :, :, half_way:half_way+1],
        #                                              method="blended_heat_map", sign="absolute_value",
        #                                              show_colorbar=True, title="Overlayed Gradient Magnitudes")
        #         dictionary['plot_batchNum' + str(b) + '_Future' + str(j * 2) + '_Raster' + str(half_way)] = fig
        #         if gradsm.shape[3]-5> 0:
        #             fig, axis = viz.visualize_image_attr(gradsm[b, :, :, half_way-1:half_way], im[b, :, :, half_way-1:half_way],
        #                                                  method="blended_heat_map", sign="absolute_value",
        #                                                  show_colorbar=True, title="Overlayed Gradient Magnitudes")
        #             dictionary['plot_batchNum' + str(b) + '_Future' + str(j * 2) + '_Raster' + str(half_way-1)] = fig
        #             fig, axis = viz.visualize_image_attr(gradsm[b, :, :, raster_len - 4:raster_len - 3],
        #                                                  im[b, :, :, raster_len - 4:raster_len - 3],
        #                                                  method="blended_heat_map", sign="absolute_value",
        #                                                  show_colorbar=True, title="Overlayed Gradient Magnitudes")
        #             dictionary['plot_batchNum' + str(b) + '_Future' + str(j * 2) + '_Raster' + str(raster_len-4)] = fig
        #         fig, axis = viz.visualize_image_attr(gradsm[b,:,:,raster_len-3:raster_len], im[b,:,:,raster_len-3:raster_len],
        #                                              method="blended_heat_map", sign="absolute_value",
        #                                              show_colorbar=True, title="Overlayed Gradient Magnitudes")
        #         dictionary['plot_batchNum'+ str(b) + '_Future' + str(j * 2) + '_RasterBackground' ] = fig
        if j == 0:
            grads_total = (grads + gradsp)
        else:
            grads_total += (grads + gradsp)
    gradsm = grads_total / (2 * future_len)
    gradsm = np.transpose(gradsm.squeeze().cpu().detach().numpy(), (0,2, 3, 1))
    for b in range(batch_num):
        fig, axis = viz.visualize_image_attr(gradsm[b, :, :, 0:1], im[b, :, :, 0:1],
                                             method="blended_heat_map", sign="absolute_value",
                                             show_colorbar=True, title="Overlayed Gradient Magnitudes",use_pyplot= False)
        dictionary['plot_batchNum' + str(b) + '_FutureTotal_Raster' + str(0)] = fig
        fig, axis = viz.visualize_image_attr(gradsm[b, :, :,half_way:half_way+1], im[b, :, :, half_way:half_way+1],
                                             method="blended_heat_map", sign="absolute_value",
                                             show_colorbar=True, title="Overlayed Gradient Magnitudes",use_pyplot= False)
        dictionary['plot_batchNum' + str(b) + '_FutureTotal_Raster' + str(half_way)] = fig
        if gradsm.shape[3] - 5 > 0:
            fig, axis = viz.visualize_image_attr(gradsm[b, :, :, half_way-1:half_way],
                                                 im[b, :, :, half_way-1:half_way],
                                                 method="blended_heat_map", sign="absolute_value",
                                                 show_colorbar=True, title="Overlayed Gradient Magnitudes",use_pyplot= False)
            dictionary[
                'plot_batchNum' + str(b) + '_FutureTotal_Raster' + str(half_way-1)] = fig
            fig, axis = viz.visualize_image_attr(gradsm[b, :, :, raster_len - 4:raster_len - 3],
                                                 im[b, :, :, raster_len - 4:raster_len - 3],
                                                 method="blended_heat_map", sign="absolute_value",
                                                 show_colorbar=True, title="Overlayed Gradient Magnitudes",use_pyplot= False)
            dictionary[
                'plot_batchNum' + str(b) + '_FutureTotal_Raster' + str(raster_len- 4)] = fig
        fig, axis = viz.visualize_image_attr(gradsm[b, :, :, raster_len - 3:raster_len],
                                             im[b, :, :, raster_len- 3:raster_len],
                                             method="blended_heat_map", sign="absolute_value",
                                             show_colorbar=True, title="Overlayed Gradient Magnitudes",use_pyplot= False)
        dictionary['plot_batchNum' + str(b) + '_FutureTotal_RasterBackground'] = fig
    return dictionary

from abc import ABC

import torch
import pytorch_lightning as pl
from typing import Optional, Union

from captum.attr import Saliency
from raster.models import BaseResnet
from raster.utils import str2bool
from l5kit.configs import load_config_data
from .utils import find_batch_extremes, draw_batch, saliency_map, filter_batch, batch_stats
from argparse import ArgumentParser


class LyftTrainerModule(pl.LightningModule, ABC):
    def __init__(
            self,
            config: Union[str, dict],
            model: Optional[Union[torch.nn.Module, str]] = None,
            optimizer: Optional[str] = ('Adam', {'lr': 1e-4}),
            scheduler: Optional[str] = None,
            extreme_k: Optional[int] = 5,
            visualization_interval: Optional[int] = 50,
            trajstat_threshold: Optional[float] = 3.,
            filter_static_history: Optional[float] = 1.,
            filter_train_batches: Optional[bool] = True
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config if isinstance(config, dict) else load_config_data(config)
        self.lr = config['train_params'].get('lr', 1e-4)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model or BaseResnet(
            in_channels=(config["model_params"]["history_num_frames"] + 1) * 2 + 3,
            out_dim=2 * config["model_params"]["future_num_frames"],
            model_type=config['model_params']['model_architecture'],
            pretrained=True,
        )
        self.criterion = lambda x, y: torch.square((x - y).view(x.shape[0], -1)).mean(1)
        self.visualize_interval = visualization_interval
        self.extreme_k = extreme_k
        self.filter_train_batches = filter_train_batches
        self.filter_static_history = filter_static_history
        self.trajectory_stat_threshold = trajstat_threshold
        if self.visualize_interval:
            self.saliency = Saliency(self)

    def forward(self, inputs, targets: torch.Tensor, target_availabilities: torch.Tensor = None, return_outputs=True):
        outputs = self.model(inputs).view(targets.shape)
        if target_availabilities is not None:
            outputs = outputs * target_availabilities.unsqueeze(-1)
        loss = self.criterion(outputs, targets)
        if return_outputs:
            return loss, outputs
        return loss

    def track_grads(self, batch, loop_name='val', batch_name=''):
        fig, grads = saliency_map(batch, self.saliency)
        self.logger.experiment.add_figure(
            f'{loop_name}/{f"{batch_name}/" if batch_name else ""}saliency', fig)
        self.logger.experiment.add_scalar(
            f'gradient/norm/{loop_name}/{f"{batch_name}/" if batch_name else ""}',
            torch.Tensor([grad.norm() for grad in grads]).mean())

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int):
        if self.filter_train_batches and not filter_batch(batch, self.filter_static_history):
            return -1
        return

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss, outputs = self(batch['image'], batch['target_positions'], batch['target_availabilities'])
        mean_loss = loss.mean()
        result = pl.TrainResult(minimize=mean_loss)
        result.log('loss/train', mean_loss, logger=True, prog_bar=True, on_epoch=True)
        result.log_dict(
            batch_stats(batch, outputs, self.trajectory_stat_threshold, 'data/train/'), logger=True,
            reduce_fx=lambda x: sum(x) / len(x), on_epoch=True
        )
        return result

    def visualize_batch(self, batch, batch_idx, loss, outputs, loop_name='val'):
        if self.visualize_interval is not None and batch_idx % self.visualize_interval == 0:
            best_batch, worst_batch = find_batch_extremes(batch, loss, outputs, self.extreme_k)
            rasterizer = self.datamodule.rasterizer

            self.logger.experiment.add_images(
                f'{loop_name}/best',
                draw_batch(rasterizer=rasterizer, batch=best_batch, outputs=best_batch['outputs']))
            self.logger.experiment.add_images(
                f'{loop_name}/worst',
                draw_batch(rasterizer=rasterizer, batch=worst_batch, outputs=worst_batch['outputs']))
            self.track_grads(best_batch, loop_name, 'best')
            self.track_grads(worst_batch, loop_name, 'worst')

    def validation_step(self, batch, batch_idx) -> pl.EvalResult:
        loss, outputs = self(batch['image'], batch['target_positions'], batch['target_availabilities'])
        mean_loss = loss.mean()
        self.visualize_batch(batch, batch_idx, loss, outputs, 'val')
        result = pl.EvalResult(checkpoint_on=mean_loss)
        result.log_dict(
            batch_stats(batch, outputs, self.trajectory_stat_threshold, 'data/val/'), logger=True,
            reduce_fx=lambda x: sum(x) / len(x),
            on_epoch=True
        )
        result.log('loss/val', mean_loss, logger=True, on_epoch=True, on_step=True)
        # result.log('val_loss', mean_loss, prog_bar=True)
        return result

    def configure_optimizers(self):
        opt_class, opt_dict = torch.optim.Adam, {'lr': self.lr}
        if self.optimizer:
            opt_class = getattr(torch.optim, self.optimizer[0])
            opt_dict = self.optimizer[-1]
        opt = opt_class(self.parameters(), **opt_dict)

        sched_class, sched_dict = torch.optim.lr_scheduler.StepLR, {'step_size': 50, 'gamma': 0.5}
        if self.scheduler:
            sched_class = getattr(torch.optim.lr_scheduler, self.scheduler[0])
            sched_dict = self.scheduler[-1]
        sched = sched_class(opt, **sched_dict)
        return [opt], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--visualization_interval', type=int, default=50,
                            help='if set to a positive value will be the interval between steps in visualization')
        parser.add_argument('--extreme_k', type=int, default=5,
                            help='count of extreme items in batch to be visualized')
        parser.add_argument('--trajstat_threshold', type=float, default=3.,
                            help='threshold in meters used to classify trajectory type')
        parser.add_argument('--filter_static_history', type=float, default=1.,
                            help='either a positive value in meters to filter static history items in batch')
        parser.add_argument('--filter_train_batches', type=str2bool, const=True, default=True, nargs='?',
                            help='whether to filter train_batches')
        return parser

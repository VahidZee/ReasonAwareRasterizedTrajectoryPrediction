import torch
import pytorch_lightning as pl
from typing import Optional, Union, Callable, Dict, Tuple, NewType, List, Any
from .models import BaseResnet
from l5kit.configs import load_config_data



class BaseTrainerModule(pl.LightningModule):
    STATIC_HISTORY_THRESHOLD = 1.

    def __init__(
            self,
            config: Union[str, dict],
            model: Optional[Union[torch.nn.Module, str]] = None,
            optimizer: Optional[str] = ('Adam', {'lr': 1e-4}),
            scheduler: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config if isinstance(config, dict) else load_config_data(config)
        self.lr = 1e-4
        self.model = model or BaseResnet(
            in_channels=(config["model_params"]["history_num_frames"] + 1) * 2 + 3,
            out_dim=2 * config["model_params"]["future_num_frames"],
            model_type=config['model_params']['model_architecture'],
            pretrained=True,
        )
        self.criterion = torch.nn.MSELoss(reduction='none')

    def forward(self, batch, batch_idx=None, apply_target_availabilities=False):
        inputs = batch["image"]
        targets = batch["target_positions"]
        outputs = self.model(inputs).view(targets.shape)
        loss = self.criterion(outputs, targets)
        if apply_target_availabilities:
            loss = loss * batch['target_availabilities'].unsqeeze(-1)
        return loss, outputs

    def on_train_batch_start(self, batch, batch_idx: int, dataloader_idx: int):
        # filter scenes with low target availabilities
        tar_idxs = (batch['target_availabilities'].sum(axis=1).data == self.config['model_params']['future_num_frames'])

        # filter scenes with low history availabilities
        his_idxs = (batch['history_availabilities'].sum(axis=1).data == self.config['model_params'][
            'history_num_frames'] + 1)

        # filter scenes with static history
        if self.config['model_params']['history_num_frames']:
            diff = batch['history_positions'][:, -1].data - batch['history_positions'][:, 0].data
            stat_idxs = (diff.norm(p=2, dim=1) > self.STATIC_HISTORY_THRESHOLD)
        else:
            stat_idxs = torch.ones(tar_idxs.shape[0], dtype=torch.bool)

        # applying filters
        idxs = (tar_idxs * his_idxs * stat_idxs).data
        if not idxs.sum():
            return -1
        for key in batch:
            batch[key] = batch[key][idxs]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss, _ = self(batch, batch_idx)
        result = pl.TrainResult(minimize=loss.mean())
        return result

    # def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
    #     return

    def validation_step(self, batch, batch_idx) -> pl.EvalResult:
        loss, outputs = self(batch, batch_idx)
        mean_loss = loss.mean()

        result = pl.EvalResult()
        result.log('val_loss', mean_loss, logger=True, prog_bar=True)
        return result

    def validation_epoch_end(self, outputs):
        # print(outputs)
        self.asghar = outputs

    def configure_optimizers(self):
        opt_class, opt_dict = torch.optim.Adam, {'lr': self.lr}
        if self.hparams.optimizer:
            opt_class = getattr(torch.optim, self.hparams.optimizer[0])
            opt_dict = self.hparams.optimizer[-1]
        opt = opt_class(self.parameters(), **opt_dict)

        sched_class, sched_dict = torch.optim.lr_scheduler.StepLR, {'step_size': 50, 'gamma': 0.5}
        if self.hparams.scheduler:
            sched_class = getattr(torch.optim.lr_scheduler, self.hparams.scheduler[0])
            sched_dict = self.hparams.scheduler[-1]
        sched = sched_class(opt, **sched_dict)
        return [opt], [sched]

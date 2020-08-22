import torch
import pytorch_lightning as pl
from typing import Optional, Union, Callable, Dict, Tuple, NewType, List

# (OptimizerModuleName, [Parameters], OptimizerArgsDict)
Identifier = NewType('Identifier', Tuple[str, Optional[Union[str, int]], Dict])


class BaseTrainerModule(pl.LightningModule):
    def __init__(self, model,
                 criterion: Optional[Union[Callable, None]] = None,
                 optimizer: Optional[Union[Identifier, List[Identifier]]] = None,
                 scheduler: Optional[Union[Identifier, List[Identifier]]] = None,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = 1e-4
        self.model = self.hparams.model
        self.criterion = self.hparams.criterion or torch.nn.MSELoss(reduction='none')

    def forward(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"]  # baseline also did unsqueeze(-1)
        targets = batch["target_positions"]
        outputs = self.model(inputs).view(targets.shape)
        loss = (self.criterion(outputs, targets) * target_availabilities).mean()
        return loss, outputs

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss, _ = self(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log('loss/train', loss, logger=True)
        return result

    def validation_step(self, batch, batch_idx) -> pl.EvalResult:
        loss, _ = self(batch, batch_idx)
        result = pl.EvalResult()
        result.log('loss/val', loss, logger=True)
        return result

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

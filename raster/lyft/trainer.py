from abc import ABC

import torch
import pytorch_lightning as pl
import typing as th

from captum.attr import Saliency
import raster.models as models
from raster.utils import KeyValue, boolify
from l5kit.configs import load_config_data

from .saliency_supervision import SaliencySupervision
from .utils import find_batch_extremes, filter_batch, neg_multi_log_likelihood
from argparse import ArgumentParser


class LyftTrainerModule(pl.LightningModule, ABC):
    def __init__(
            self,
            config: dict,
            model: str = 'Resnet',
            modes: int = 1,
            optimizer: str = 'Adam',
            optimizer_dict: th.Optional[dict] = None,
            lr: float = 1e-4,
            scheduler: th.Optional[str] = None,
            scheduler_dict: th.Optional[dict] = None,
            saliency_factor: float = 0.,
            saliency_intrest: str = 'simple',
            saliency_dict: th.Optional[dict] = None,
            pgd_iters: int = 0,
            pgd_alpha: float = 0.01,
            pgd_random_start: bool = False,
            pgd_eps_vehicles: float = 0.4,
            pgd_eps_semantics: float = 0.15625,
            track_grad: bool = False,
            **kwargs
    ):
        super().__init__()

        hparams = dict(
            config=config, optimizer=optimizer, scheduler=scheduler, model=model, lr=lr, modes=modes,
            saliency_factor=saliency_factor, saliency_intrest=saliency_intrest, pgd_iters=pgd_iters,
            pgd_alpha=pgd_alpha, pgd_random_start=pgd_random_start, pgd_eps_vehicles=pgd_eps_vehicles,
            pgd_eps_semantics=pgd_eps_semantics)
        defhparams = list(hparams.keys())
        for name, _dict in [('scheduler', scheduler_dict), ('optimizer', optimizer_dict), ('saliency', saliency_dict)]:
            for item in _dict if _dict else dict():
                hparams[f'{name}_{item}'] = _dict[item]
        self.save_hyperparameters(hparams)
        scheduler_dict = scheduler_dict or dict()
        optimizer_dict = optimizer_dict or dict()
        saliency_dict = saliency_dict or dict()
        model_kwargs = dict()
        for key in self.hparams:
            if key not in defhparams:
                kwargs[key] = self.hparams[key]
                if key.startswith('scheduler_'):
                    scheduler_dict[key.replace("scheduler_", "")] = self.hparams[key]
                elif key.startswith('optimizer_'):
                    optimizer_dict[key.replace("optimizer_", "")] = self.hparams[key]
                elif key.startswith('saliency_'):
                    saliency_dict[key.replace("saliency_", "")] = self.hparams[key]
                else:
                    model_kwargs[key] = self.hparams[key]
        self.model = getattr(models, self.hparams.model)(config=config, modes=modes, **model_kwargs)
        self.config = self.hparams.config
        self.modes = self.hparams.modes
        self.saliency_factor = self.hparams.saliency_factor
        self.saliency = SaliencySupervision(
            self.hparams.saliency_intrest, **saliency_dict) if self.saliency_factor else None
        self.pgd_iters = self.hparams.pgd_iters
        self.pgd_random_start = self.hparams.pgd_random_start
        self.pgd_alpha = self.hparams.pgd_alpha
        self.pgd_eps_semantics = self.hparams.pgd_eps_semantics
        self.pgd_eps_vehicles = self.hparams.pgd_eps_vehicles
        self.lr = self.hparams.lr
        self.optimizer = self.hparams.optimizer
        self.optimizer_dict = optimizer_dict
        self.scheduler = self.hparams.scheduler
        self.scheduler_dict = scheduler_dict
        self.track_grad = track_grad

    def pgd_attack(self, inputs, outputs, target_availabilities=None, return_loss=True):
        if self.pgd_random_start:
            delta = (torch.rand_like(inputs) - 0.5) * 2
            for (s_channel, end_channel), eps in [
                ((- 3, None), self.pgd_eps_semantics),
                ((0, - 3), self.pgd_eps_vehicles)
            ]:
                delta[:, s_channel:end_channel] *= eps
            delta.requires_grad = True
        else:
            delta = torch.zeros_like(inputs, requires_grad=True)

        # freezing model params
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        for i in range(self.pgd_iters):
            loss = neg_multi_log_likelihood(outputs, *self.model((inputs.detach() + delta).clamp(0, 1.)),
                                            target_availabilities).mean()
            if i == 0 and return_loss:
                init_loss = loss.detach()
            loss.backward()
            delta.data = delta.data + self.pgd_alpha * delta.grad.detach().sign()
            for (s_channel, end_channel), eps in [
                ((- 3, None), self.pgd_eps_semantics),
                ((0, - 3), self.pgd_eps_vehicles)
            ]:
                delta.data[:, s_channel:end_channel].clamp_(-eps, eps)
            delta.grad.zero_()
        if return_loss:
            final_loss = neg_multi_log_likelihood(outputs, *self.model((inputs + delta).detach().clamp(0, 1.)),
                                                  target_availabilities).mean()

        # unfreezing model
        for param in self.parameters():
            param.requires_grad = True
        self.train()
        if return_loss:
            return (inputs.detach() + delta.detach()).clamp(0, 1.), init_loss, final_loss
        return (inputs.detach() + delta.detach()).clamp(0, 1.)

    def forward(self, inputs, targets: torch.Tensor, target_availabilities: torch.Tensor = None, return_results=True,
                grad_enabled=True, attack=True):
        torch.set_grad_enabled(grad_enabled)
        res = dict()
        if self.pgd_iters and attack:
            inputs, init_loss, final_loss = self.pgd_attack(inputs, targets, target_availabilities, return_loss=True)
            res['attack/init_loss'] = init_loss
            res['attack/final_loss'] = final_loss
        inputs.requires_grad = bool(self.saliency_factor) or self.track_grad
        pred, conf = self.model(inputs)
        nll = neg_multi_log_likelihood(targets, pred, conf, target_availabilities)
        if (self.saliency_factor or self.track_grad) and grad_enabled:
            grads = torch.autograd.grad(nll.unbind(), inputs, create_graph=bool(self.saliency_factor))[0]
            res['grads/semantics'] = grads.data[:, -3:].abs().sum()
            res['grads/vehicles'] = grads.data[:, :-3].abs().sum()
            res['grads/total'] = res['grads/semantics'] + res['grads/vehicles']
        if self.saliency_factor and grad_enabled:
            sal_res = self.saliency(grads)
            res['loss'] = ((1 - sal_res) * self.saliency_factor * nll.detach() + nll).mean()
            res['saliency'] = sal_res.mean()
            res['nll'] = nll.mean()
        else:
            res['loss'] = nll.mean()
        if return_results:
            return res
        return res['loss']

    def step(self, batch, batch_idx, optimizer_idx=None, name='train'):
        is_val = name == 'val'
        result = self(batch['image'], batch['target_positions'], batch.get('target_availabilities', None),
                      return_results=True, attack=not is_val)
        for item, value in result.items():
            self.log(f'{item}/{name}', value.mean(), on_step=not is_val, on_epoch=is_val, logger=True, sync_dist=True)
        if not is_val:
            return result['loss']

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.step(batch, batch_idx, optimizer_idx, name='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, name='val')

    def configure_optimizers(self):
        opt_class, opt_dict = torch.optim.Adam, {'lr': self.lr}
        if self.optimizer:
            opt_class = getattr(torch.optim, self.optimizer)
            opt_dict = self.optimizer_dict or dict()
            opt_dict['lr'] = self.lr
        opt = opt_class(self.parameters(), **opt_dict)
        if self.hparams.scheduler is None:
            return opt
        sched_class, sched_dict = torch.optim.lr_scheduler.StepLR, {'step_size': 50, 'gamma': 0.5}
        if self.scheduler:
            sched_class = getattr(torch.optim.lr_scheduler, self.scheduler)
            sched_dict = self.scheduler_dict
        sched_dict = sched_dict or dict()
        sched = sched_class(opt, **sched_dict)
        return [opt], [sched]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model', type=str, default='Resnet', help='model architecture class to use')
        parser.add_argument('--modes', type=int, default=1, help='number of modes of model prediction')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use')
        parser.add_argument('--optimizer-dict', nargs='*', action=KeyValue, help='additional optimizer specific args')
        parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        parser.add_argument('--scheduler', type=str, default=None, help='scheduler to use')
        parser.add_argument('--scheduler-dict', nargs='*', action=KeyValue, help='additional scheduler specific args')
        parser.add_argument('--pgd-mode', type=str, default='loss', help='pgd attack mode [loss/input]')
        parser.add_argument('--pgd-iters', type=int, default=0, help='pgd attack number of iterations')
        parser.add_argument('--pgd-random-start', type=boolify, default=False,
                            help='whether to use a random point for adversary search during pgd attack')
        parser.add_argument('--pgd-alpha', type=float, default=1e-2, help='pgd attack alpha value')
        parser.add_argument('--pgd-eps-vehicles', type=float, default=0.4,
                            help='epsilon bound for pgd attack on vehicle layers')
        parser.add_argument('--pgd-eps-semantics', type=float, default=0.15625,
                            help='epsilon bound for pgd attack on semantic layers')
        parser.add_argument('--saliency-factor', type=float, default=1e-4, help='saliency supervision factor')
        parser.add_argument('--saliency-intrest', type=str, default='simple',
                            help='intrest region calculation for saliency supervision')
        parser.add_argument('--saliency-dict', nargs='*', action=KeyValue,
                            help='additional saliency supervision specific args')
        parser.add_argument('--track-grad', type=boolify, default=False,
                            help='whether to log grad norms')
        parser.add_argument('--kwargs', nargs='*', default=dict(), action=KeyValue,
                            help='additional model specific args')
        return parser

from abc import ABC
import pytorch_lightning as pl
import importlib


class MobilenetV2(pl.LightningModule, ABC):
    def __init__(self, in_channels, out_dim, model_type: str = 'mobilenet_v2', pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        from torch.nn import Linear, Conv2d  # in case they weren't imported
        self.model = getattr(importlib.import_module('torchvision.models'), self.hparams.model_type)(
            pretrained=self.hparams.pretrained)
        if self.model.features[0][0].in_channels != self.hparams.in_channels:
            old_weight = self.model.features[0][0].weight
            old_bias = self.model.features[0][0].bias
            if isinstance(self.hparams.in_channels, int):
                self.model.features[0][0] = Conv2d(
                    in_channels=self.hparams.in_channels, out_channels=self.model.features[0][0].out_channels,
                    stride=self.model.features[0][0].stride,
                    bias=self.model.features[0][0].bias is not None, kernel_size=self.model.features[0][0].kernel_size,
                    padding=self.model.features[0][0].padding,
                    padding_mode=self.model.features[0][0].padding_mode, dilation=self.model.features[0][0].dilation)
                self.model.features[0][0].weight[:, :min(self.hparams.in_channels, old_weight.shape[1]), :, :].data = \
                    old_weight[:, :min(self.hparams.in_channels, old_weight.shape[1]), :, :].data
            else:
                self.model.features[0][0] = Conv2d(
                    in_channels=self.hparams.in_channels['out_dim'],
                    out_channels=self.model.features[0][0].out_channels,
                    stride=self.model.features[0][0].stride,
                    bias=self.model.features[0][0].bias is not None, kernel_size=self.model.features[0][0].kernel_size,
                    padding=self.model.features[0][0].padding,
                    padding_mode=self.model.features[0][0].padding_mode, dilation=self.model.features[0][0].dilation)
                self.model.features[0][0].weight[:, :min(
                    self.hparams.in_channels['out_dim'], old_weight.shape[1]), :, :].data = \
                    old_weight[:, :min(self.hparams.in_channels['out_dim'], old_weight.shape[1]), :, :].data
            if old_bias is not None:
                self.model.features[0][0].bias.data = old_bias.data
            if self.model.classifier[1].out_features != self.hparams.out_dim:
                old_weight = self.model.classifier[1].weight
                old_bias = self.model.classifier[1].bias
                self.model.classifier[1] = Linear(self.model.classifier[1].in_features, self.hparams.out_dim,
                                                  bias=self.model.classifier[1].bias is not None)
                self.model.classifier[1].weight[:min(old_weight.shape[0], self.hparams.out_dim), :].data = \
                    old_weight[:min(old_weight.shape[0], self.hparams.out_dim), :].data
                if old_bias is not None:
                    self.model.classifier[1].bias[:min(self.hparams.out_dim, old_bias.shape[0])].data = \
                        old_bias[:min(self.hparams.out_dim, old_bias.shape[0])].data

    def forward(self, x):
        return self.model(x)

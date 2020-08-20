import pytorch_lightning as pl
import importlib


class BaseResnet(pl.LightningModule):
    def __init__(self, input_shape, num_classes, model_type: str = 'resnet50', pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        from torch.nn import Linear, Conv2d  # in case they weren't imported
        self.model = getattr(importlib.import_module('torchvision.models'), self.hparams.model_type)(
            pretrained=self.hparams.pretrained)
        if self.model.conv1.in_channels != self.hparams.input_shape[0]:
            old_weight = self.model.conv1.weight
            old_bias = self.model.conv1.bias
            self.model.conv1 = Conv2d(
                in_channels=self.hparams.input_shape[0], out_channels=self.model.conv1.out_channels,
                stride=self.model.conv1.stride,
                bias=self.model.conv1.bias is not None, kernel_size=self.model.conv1.kernel_size,
                padding=self.model.conv1.padding,
                padding_mode=self.model.conv1.padding_mode, dilation=self.model.conv1.dilation)
            self.model.conv1.weight[:, :min(self.hparams.input_shape[0], old_weight.shape[1]), :, :].data = \
                old_weight[:, :min(self.hparams.input_shape[0], old_weight.shape[1]), :, :].data
            if old_bias is not None:
                self.model.conv1.bias.data = old_bias.data
            if self.model.fc.out_features != num_classes:
                old_weight = self.model.fc.weight
                old_bias = self.model.fc.bias
                self.model.fc = Linear(self.model.fc.in_features, num_classes, bias=self.model.fc.bias is not None)
                self.model.fc.weight[:min(old_weight.shape[0], self.hparams.num_classes), :].data = \
                    old_weight[:min(old_weight.shape[0], self.hparams.num_classes), :].data
                if old_bias is not None:
                    self.model.fc.bias[:min(self.hparams.num_classes, old_bias.shape[0])].data = \
                        old_bias[:min(self.hparams.num_classes, old_bias.shape[0])].data

    def forward(self, x):
        return self.model(x)

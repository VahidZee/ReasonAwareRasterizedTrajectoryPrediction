from abc import ABC
import torch
import torchvision as tv

from raster.models.template import RasterModel


class Resnet(RasterModel):
    def __init__(self, config: dict, modes=1, model_type: str = 'resnet50', pretrained=False):
        super().__init__(config=config, modes=modes)
        self.model = getattr(tv.models, model_type)(
            pretrained=pretrained)
        if self.model.conv1.in_channels != self.in_channels:
            old_weight = self.model.conv1.weight
            old_bias = self.model.conv1.bias
            self.model.conv1 = torch.nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.model.conv1.out_channels,
                stride=self.model.conv1.stride,
                bias=self.model.conv1.bias is not None, kernel_size=self.model.conv1.kernel_size,
                padding=self.model.conv1.padding,
                padding_mode=self.model.conv1.padding_mode, dilation=self.model.conv1.dilation)
            self.model.conv1.weight[:, :min(self.in_channels, old_weight.shape[1]), :, :].data = \
                old_weight[:, :min(self.in_channels, old_weight.shape[1]), :, :].data
            if old_bias is not None:
                self.model.conv1.bias.data = old_bias.data
            if self.model.fc.out_features != self.out_dim:
                old_weight = self.model.fc.weight
                old_bias = self.model.fc.bias
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.out_dim,
                                                bias=self.model.fc.bias is not None)
                self.model.fc.weight[:min(old_weight.shape[0], self.out_dim), :].data = \
                    old_weight[:min(old_weight.shape[0], self.out_dim), :].data
                if old_bias is not None:
                    self.model.fc.bias[:min(self.out_dim, old_bias.shape[0])].data = \
                        old_bias[:min(self.out_dim, old_bias.shape[0])].data

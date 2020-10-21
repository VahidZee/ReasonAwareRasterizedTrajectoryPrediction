from abc import ABC
import torch


class RasterModel(torch.nn.Module, ABC):
    def __init__(self, config: dict, modes: int = 1):
        super().__init__()
        self.modes = modes
        self.in_channels = (config["model_params"]["history_num_frames"] + 1) * 2 + 3
        self.future_len = config["model_params"]["future_num_frames"] // config["model_params"]["future_step_size"]
        self.num_preds = self.modes * 2 * self.future_len
        self.out_dim = self.num_preds + (self.modes if self.modes != 1 else 0)

    def forward(self, x):
        res = self.model(x)
        bs = x.shape[0]
        if self.modes != 1:
            pred, conf = torch.split(res, self.num_preds, dim=1)
            pred = pred.view(bs, self.modes, self.future_len, 2)
            conf = torch.softmax(conf, dim=1)
            return pred, conf
        return res.view(bs, 1, self.future_len, 2), res.new_ones((bs, 1))

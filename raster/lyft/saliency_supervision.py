import torch


class SaliencySupervision(torch.nn.Module):
    def __init__(self, intrest='simple', **kwargs):
        super().__init__()
        self.intrest = intrest
        for item, value in kwargs:
            setattr(self, item, value)
        self.intrest_func = getattr(self, self.intrest)

    def forward(self, grads):
        intrest = self.intrest_func(grads)
        intrest_sum = intrest.abs().sum(dim=[1, 2, 3])
        total = grads.abs().sum(dim=[1, 2, 3])
        return torch.true_divide(intrest_sum, total)

    def simple(self, grads):
        try:
            xs, xe = self.xs, self.xe
            ys, ye = self.ys, self.ye
        except:
            xs, xe = 0.15, 0.6
            ys, ye = 0.35, 0.65
        xs, xe = int(xs * grads.shape[3]), int(xe * grads.shape[3])
        ys, ye = int(ys * grads.shape[2]), int(ye * grads.shape[2])
        return grads[:, :, ys:ye, xs:xe]

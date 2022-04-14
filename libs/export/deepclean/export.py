import torch


class PrePostDeepClean(torch.nn.Module):
    def __init__(self, deepclean: torch.nn.Module):
        super().__init__(self)

        self.deepclean = deepclean

        self.input_shift = self.add_processing_param(0)
        self.input_scale = self.add_processing_param(1)

        self.output_shift = self.add_processing_param(0)
        self.output_scale = self.add_processing_param(1)

    def add_processing_param(self, value: float):
        return torch.nn.Parameter([value], requires_grad=False).to(
            self.deepclean.device
        )

    def set_param_value(self, param: torch.nn.Parameter, value):
        param.data = torch.Tensor(
            value, dtype=torch.float32, device=param.device
        )

    def fit(self, X, y):
        self.set_param_value(self.input_shift, X.mean(axis=1, keepdims=True))
        self.set_param_value(self.input_scale, X.std(axis=1, keepdims=True))

        self.set_param_value(self.output_shift, y.mean())
        self.set_param_value(self.output_scale, y.std())

    def forward(self, x):
        x = (x - self.input_shift) / self.input_scale
        x = self.deepclean(x)
        x = self.output_scale * x + self.output_shift
        return x

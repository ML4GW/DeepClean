import torch


class PrePostDeepClean(torch.nn.Module):
    def __init__(self, deepclean: torch.nn.Module):
        super().__init__()

        self.deepclean = deepclean
        device = next(deepclean.parameters()).device

        self.input_shift = self.add_processing_param(0, device)
        self.input_scale = self.add_processing_param(1, device)

        self.output_shift = self.add_processing_param(0, device)
        self.output_scale = self.add_processing_param(1, device)

    def add_processing_param(self, value: float, device: str):
        return torch.nn.Parameter(
            torch.Tensor([value]).to(device), requires_grad=False
        )

    def set_param_value(self, param: torch.nn.Parameter, value):
        value = value.astype("float32")
        try:
            value = torch.Tensor(value)
        except TypeError:
            value = torch.Tensor([value])
        param.data = value.to(param.device)

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

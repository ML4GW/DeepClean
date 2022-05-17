import torch


def _make_tensor(value, device):
    try:
        value = torch.Tensor(value)
    except TypeError:
        value = torch.Tensor([value])
    return value.type(torch.float32).to(device)


class PrePostDeepClean(torch.nn.Module):
    def __init__(self, deepclean: torch.nn.Module):
        super().__init__()

        try:
            self.num_witnesses = deepclean.num_witnesses
        except AttributeError:
            raise ValueError(
                f"DeepClean architecture {deepclean} has "
                "no attribute 'num_witnesses', can't initialize "
                "preprocessing model."
            )

        self.deepclean = deepclean
        device = next(deepclean.parameters()).device

        self.input_shift = self.add_processing_param(
            [[0]] * self.num_witnesses, device
        )
        self.input_scale = self.add_processing_param(
            [[1]] * self.num_witnesses, device
        )

        self.output_shift = self.add_processing_param(0.0, device)
        self.output_scale = self.add_processing_param(1.0, device)

    def add_processing_param(self, value: float, device: str):
        return torch.nn.Parameter(
            _make_tensor(value, device), requires_grad=False
        )

    def fit(self, X, y):
        if X.shape[0] != self.num_witnesses:
            raise ValueError(
                "Can't fit PrePostDeepClean model with "
                "{} witness channels to data with {} channels".format(
                    self.num_witnesses, X.shape[0]
                )
            )

        device = self.input_shift.device
        Xmean = X.mean(axis=1, keepdims=True)
        Xstd = X.std(axis=1, keepdims=True)
        self.input_shift.data = _make_tensor(Xmean, device)
        self.input_scale.data = _make_tensor(Xstd, device)

        self.output_shift.data = _make_tensor(y.mean(), device)
        self.output_scale.data = _make_tensor(y.std(), device)

    def forward(self, x):
        x = (x - self.input_shift) / self.input_scale
        x = self.deepclean(x)
        x = self.output_scale * x + self.output_shift
        return x

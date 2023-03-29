import torch
from ml4gw.transforms import ChannelWiseScaler


class Preprocessor(torch.nn.Module):
    def __init__(self, scaler: ChannelWiseScaler) -> None:
        super().__init__()
        self.scaler = scaler

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.scaler(X)


class Postprocessor(torch.nn.Module):
    def __init__(self, scaler: ChannelWiseScaler) -> None:
        super().__init__()
        self.scaler = scaler

    def forward(self, X: torch.Tensor):
        return self.scaler(X, reverse=True)


class DeepClean(torch.nn.Module):
    def __init__(
        self,
        preprocessor: ChannelWiseScaler,
        deepclean: torch.nn.Module,
        postprocessor: ChannelWiseScaler
    ) -> None:
        super().__init__()
        self.preprocessor = Preprocessor(preprocessor)
        self.deepclean = deepclean
        self.postprocessor = Postprocessor(postprocessor)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.preprocessor(X)
        X = self.deepclean(X)
        X = self.postprocessor(X)
        return X
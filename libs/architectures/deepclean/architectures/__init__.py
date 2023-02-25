import inspect

from .convolutional_autoencoder import DeepCleanAE


def _wrap_arch(arch):
    def func(*args, **kwargs):
        def f(num_channels):
            return arch(num_channels, *args, **kwargs)

        return f

    params = inspect.signature(arch).parameters
    params = list(params.values())[1:]
    func.__signature__ = inspect.Signature(params)
    return func


architectures = {"autoencoder": _wrap_arch(DeepCleanAE)}

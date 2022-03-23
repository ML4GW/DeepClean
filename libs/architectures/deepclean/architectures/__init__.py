import inspect

from .convolutional_autoencoder import DeepCleanAE

architectures = {"autoencoder": DeepCleanAE}


def get_arch_fns(fn, fn_kwargs={}):
    """Create functions for network architectures

    For each network architecture, create a function which
    exposes architecture parameters as arguments and returns
    the output of the passed function `fn` called with
    the keyword arguments `fn_kwargs` and an argument
    `architecture` which is itself a function that takes
    as input the input shape to the network, and returns
    the compiled architecture.

    As an example:
    ```python
    import argparse
    from deepclean.architectures import get_arch_fns

    def train(architecture, learning_rate, batch_size):
        network = architecture(input_shape=21)
        # do some training here
        return

    # instantiate train_kwargs now, then update
    # in-place later so that each arch_fn calls
    # `train` with some command line arguments
    train_kwargs = {}
    arch_fns = get_arch_fns(train, train_kwargs)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--arch", choices=tuple(arch_fns), type=str)
        args = vars(parser.parse_args())

        arch = args.pop("arch")
        fn = arch_fns[arch]
        train_kwargs.update(args)
        fn()
    ```

    The intended use case for this is for more complex
    model architectures which may require different
    sets of arguments, so that they can be simply
    implemented with the same training function.
    """

    arch_fns = {}
    for name, arch in architectures.items():

        def arch_fn(**arch_kwargs):
            # create a function which only takes the input
            # shape as an argument and instantiates a network
            # based on the architecture with that shape and
            # the remaining kwargs
            def get_arch(input_shape):
                return arch(input_shape, **arch_kwargs)

            # pass the function to `fn` as a kwarg,
            # then run `fn` with all the passed kwargs.
            fn_kwargs["architecture"] = get_arch
            return fn(**fn_kwargs)

        # now add all the architecture parameters other
        # than the first, which is assumed to be some
        # form of input shape, to the `arch_fn` we
        # just created via the __signature__ attribute
        params = []
        for i, param in enumerate(inspect.signature(arch).parameters.values()):
            if i > 0:
                params.append(param)

        arch_fn.__signature__ = inspect.Signature(parameters=params)
        arch_fn.__name__ = name
        arch_fns[name] = arch_fn
    return arch_fns


def architecturize(f):
    """
    Wrap a function so that if it's called without
    any arguments, it will parse arguments from the
    command line with a network architecture name
    as a positional parameter with its own subparsers.

    For example, a script that looks like

    ```python
    from typing import Callable
    from training_library import do_some_training
    from deepclean.architectures import architecturize


    @architecturize
    def my_func(architecture: Callable, learning_rate: float, batch_size: int):
        num_witness_channels = 8
        network = architecture(num_witness_channels)
        do_some_training(network, learning_rate, batch_size)


    if __name__ == "__main__":
        my_func()
    ```

    can be executed from the command line like

    ```console
    python my_script.py --learning-rate 1e-3 --batch-size 32 autoencoder
    ```

    and the wrapper will take care of mapping `"autoencoder"` to the
    corresponding architecture function which maps from an input
    shape to an initialized `torch.nn.Module`.
    """

    # doing the unthinkable and putting this import
    # here until I decide what I really want to do
    # with this function
    from hermes.typeo import typeo

    f_kwargs = {}
    arch_fns = get_arch_fns(f, f_kwargs)

    def wrapper(**kwargs):
        f_kwargs.update(kwargs)

    f_params = inspect.signature(f).parameters.values()
    parameters = [p for p in f_params if p.name != "architecture"]
    wrapper.__signature__ = inspect.Signature(parameters)
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__

    return typeo(wrapper, **arch_fns)

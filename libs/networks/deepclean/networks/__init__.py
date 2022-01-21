import inspect

from .convolutional_autoencoder import DeepCleanAE

architectures = {"autoencoder": DeepCleanAE}


def get_network_fns(fn, fn_kwargs={}):
    """Create functions for network architectures

    For each network architecture, create a function which
    exposes network parameters as arguments and returns
    the output of the passed function `fn` called with
    the keyword arguments `fn_kwargs` and an argument
    `architecture` which is itself a function that takes
    as input the input shape to the network, and returns
    the compiled architecture.

    As an example:
    ```python
    import argparse
    from deepclean.networks import get_network_fns

    def train(architecture, learning_rate, batch_size):
        network = architecture(input_shape=21)
        # do some training here
        return

    # instantiate train_kwargs now, then update
    # in-place later so that each network_fn calls
    # `train` with some command line arguments
    train_kwargs = {}
    network_fns = get_network_fns(train, train_kwargs)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning-rate", type=float)
        parser.add_argument("--batch-size", type=int)
        parser.add_argument("--arch", choices=tuple(network_fns), type=str)
        args = vars(parser.parse_args())

        arch = args.pop("arch")
        fn = network_fns[arch]
        train_kwargs.update(args)
        fn()
    ```

    The intended use case for this is for more complex
    model architectures which may require different
    sets of arguments, so that they can be simply
    implemented with the same training function.
    """

    network_fns = {}
    for name, arch in architectures.items():

        def network_fn(**arch_kwargs):
            # create a function which only takes the input
            # shape as an argument and instantiates the
            # network with that shape and remaining kwargs
            def get_network(input_shape):
                return arch(input_shape, **arch_kwargs)

            # pass the function to `fn` as a kwarg,
            # then run `fn` with all the passed kwargs.
            fn_kwargs["architecture"] = get_network
            return fn(**fn_kwargs)

        # now add all the architecture parameters other
        # than the first, which is assumed to be some
        # form of input shape, to the `network_fn` we
        # just created via the __signature__ attribute
        params = []
        for i, param in enumerate(inspect.signature(arch).parameters.values()):
            if i > 0:
                params.append(param)

        network_fn.__signature__ = inspect.Signature(parameters=params)
        network_fn.__name__ = name
        network_fns[name] = network_fn
    return network_fns

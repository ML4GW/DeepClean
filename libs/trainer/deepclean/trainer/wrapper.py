import inspect

from hermes.typeo import typeo

from deepclean.networks import networks
from deepclean.trainer.trainer import train


def _configure_wrapper(f, wrapper):
    f_sig = inspect.signature(f)
    train_sig = inspect.signature(train)

    parameters = []
    optional_parameters = []
    for p in f_sig.parameters.values():
        if p.name == "kwargs":
            continue
        elif p.default is inspect.Parameter.empty:
            parameters.append(p)
        else:
            optional_parameters.append(p)

    for param in train_sig.parameters.values():
        if (
            param.name not in ("X", "y", "valid_data", "architecture")
            and param.name not in f_sig.parameters
        ):
            parameters.append(param)
    parameters = parameters + optional_parameters

    wrapper.__signature__ = inspect.Signature(parameters=parameters)
    wrapper.__name__ = f.__name__

    train_doc = train.__doc__.split("Args:")[1]
    try:
        wrapper.__doc__ = f.__doc__ + train_doc
    except TypeError:
        pass


def _make_network_fns(train_kwargs):
    network_fns = {}
    for name, arch in networks.items():

        def network_fn(**kwargs):
            def get_network(input_shape):
                return arch(input_shape, **kwargs)

            train_kwargs["architecture"] = get_network
            return train(**train_kwargs)

        params = []
        for i, param in enumerate(inspect.signature(arch).parameters.values()):
            if i > 0:
                params.append(param)

        network_fn.__signature__ = inspect.Signature(parameters=params)
        network_fn.__name__ = name
        network_fns[name] = network_fn
    return network_fns


def make_cmd_line_fn(f):
    train_kwargs = {}
    network_fns = _make_network_fns(train_kwargs)
    train_signature = inspect.signature(train)

    def wrapper(*args, **kwargs):
        data = f(*args, **kwargs)

        for p, v in zip(inspect.signature(f).parameters, args):
            if p in train_signature.parameters:
                train_kwargs[p] = v

        for k, v in kwargs.items():
            if k in train_signature.parameters:
                train_kwargs[k] = v

        if len(data) == 2:
            X, y = data
            valid_X = valid_y = None
        elif len(data) == 4:
            X, y, valid_X, valid_y = data
        else:
            raise ValueError(
                "Can't process {} elements returned by function {}".format(
                    len(data), f.__name__
                )
            )

        train_kwargs["X"] = X
        train_kwargs["y"] = y
        if valid_X is not None:
            train_kwargs["valid_data"] = (valid_X, valid_y)

        if "arch" in kwargs:
            try:
                network_fn = network_fns[kwargs["arch"]]
            except KeyError:
                raise ValueError(
                    "No network architecture named " + kwargs["arch"]
                )

            arch_kwargs = {}
            arch_sig = inspect.signature(network_fn)
            for k, v in kwargs.items():
                if k in arch_sig.parameters:
                    arch_kwargs[k] = v

            result = network_fn(**arch_kwargs)
        else:
            result = data

        return result

    _configure_wrapper(f, wrapper)
    return typeo(wrapper, **network_fns)

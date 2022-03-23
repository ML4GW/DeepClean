import inspect

from hermes.typeo import typeo
from hermes.typeo.typeo import _parse_doc, _parse_help

from deepclean.architecture import get_arch_fns
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

    _, train_args = _parse_doc(train)
    f_doc, f_args = _parse_doc(f)

    wrapper_args = ""
    for p in parameters:
        for args in [f_args, train_args]:
            doc_str = _parse_help(args, p.name)
            if doc_str:
                break
        else:
            continue

        wrapper_args += "\n" + " " * 8 + p.name + ":\n"
        wrapper_args += " " * 12 + doc_str
    wrapper.__doc__ = f_doc + "\n" + " " * 4 + "Args:\n" + wrapper_args


def trainify(f):
    """Turn a data-generating function into a command line trainer

    Wraps the function `f`, which is assumed to generate training
    and validation data, so that this data gets passed to
    `deepclean.trainer.trainer.train`, but in such a way that
    `f` can be called from the command line with all of the arguments
    to `deepclean.trainer.trainer.train`, with the network architecture
    as a positional parameter and its arguments as additional parameters
    after that.
    """

    # initialize our training kwargs now and use them to
    # create the wrapper network functions. Populate these
    # args later within our wrapper function in-place
    train_kwargs = {}
    arch_fns = get_arch_fns(train, train_kwargs)
    train_signature = inspect.signature(train)

    def wrapper(*args, **kwargs):
        # use the passed function `f` to generate data
        data = f(*args, **kwargs)

        # pass any args passed to this wrapper that
        # `train` needs into the `train_kwargs` dictionary
        for p, v in zip(inspect.signature(f).parameters, args):
            if p in train_signature.parameters:
                train_kwargs[p] = v

        # do the same for any kwargs that were passed here
        for k, v in kwargs.items():
            if k in train_signature.parameters:
                train_kwargs[k] = v

        # parse out the data that was returned from
        # `f`, checking if validation data was included
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

        # add in the parsed data to our training kwargs
        train_kwargs["X"] = X
        train_kwargs["y"] = y
        if valid_X is not None:
            train_kwargs["valid_data"] = (valid_X, valid_y)

        # allow wrapper functionality to be utilized if
        # `f` is called with an "arch" parameter
        if "arch" in kwargs:
            try:
                arch_fn = arch_fns[kwargs["arch"]]
            except KeyError:
                raise ValueError(
                    "No network architecture named " + kwargs["arch"]
                )

            # grab any architecture-specific parameters from
            # the arguments passed to `f`
            arch_kwargs = {}
            arch_sig = inspect.signature(arch_fn)
            for k, v in kwargs.items():
                if k in arch_sig.parameters:
                    arch_kwargs[k] = v

            # run the architecture function, which will implicitly
            # call deepclean.trainer.trainer.train under the
            # hood with the arguments we populated into
            # `train_kwargs`
            result = arch_fn(**arch_kwargs)
        else:
            # otherwise just return the data, equivalent
            # to running `f` without any wrapper functionality
            result = data

        return result

    # create the appropriate signature, name, and documentation
    # for the wrapper function we just created
    _configure_wrapper(f, wrapper)

    # wrap this wrapper using typeo, so that it
    # can be called from the command line exposing
    # all training and architecture arguments. Each
    # network architecture will be exposed as a
    # subcommand with its own arguments
    return typeo(wrapper, **arch_fns)

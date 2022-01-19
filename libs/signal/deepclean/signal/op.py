import pickle
from typing import List


class Op:
    def __call__(self, x, **kwargs):
        return x

    def fit(self, X, **kwargs):
        return

    def __rshift__(self, other):
        if isinstance(other, Pipeline):
            pipeline = Pipeline([self] + other.ops)
        elif isinstance(other, Op):
            pipeline = Pipeline([self, other])
        else:
            raise TypeError(
                "Can't pipe output of Op {} to type {}".format(
                    self.__class__.__name__, type(other)
                )
            )
        return pipeline

    def write(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)


class Pipeline(Op):
    def __init__(self, ops: List[Op]):
        self.ops = ops

    def __call__(self, x, **kwargs):
        for op in self.ops:
            x = op(x)
        return x

    def fit(self, X, **kwargs):
        for op in self.ops:
            op.fit(X, **kwargs)
            X = op(X)

    def __rshift__(self, other):
        if isinstance(other, Pipeline):
            pipeline = Pipeline(self.ops + other.ops)
        elif isinstance(other, Op):
            pipeline = Pipeline(self.ops + [other])
        else:
            raise TypeError(
                "Can't pipe output of Pipeline to type {}".format(
                    type(other)
                )
            )
        return pipeline

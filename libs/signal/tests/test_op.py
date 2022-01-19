import os
import pickle

import pytest

from deepclean.signal.op import Op, Pipeline


class AddOne(Op):
    def __call__(self, x):
        return x + 1


class ParamOp(Op):
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return x + self.value


@pytest.fixture(scope="session")
def fname():
    yield "op.pkl"
    os.remove("op.pkl")


def test_op(fname):
    op = AddOne()
    assert op(4) == 5

    op.write(fname)
    op = AddOne.load(fname)
    assert op(4) == 5

    op2 = ParamOp(3)
    assert op2(4) == 7

    op2.write(fname)
    op2 = ParamOp.load(fname)
    assert op2(4) == 7

    pipeline = op >> op2
    assert pipeline(4) == 8

    pipeline.write(fname)
    pipeline = Pipeline.load(fname)
    assert pipeline(4) == 8

    op3 = AddOne()
    pipeline2 = pipeline >> op3
    assert pipeline2(4) == 9

    pipeline2.write(fname)
    pipeline2 = Pipeline.load(fname)
    assert pipeline2(4) == 9

    pipeline3 = op3 >> pipeline2
    assert pipeline3(4) == 10

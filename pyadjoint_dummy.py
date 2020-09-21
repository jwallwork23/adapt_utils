from firedrake import Function

import numpy as np

import adapt_utils.optimisation as opt


__all__ = ["stop_annotating", "Control", "taylor_test"]


class stop_annotating(object):
    """
    Dummy `stop_annotating` class for consistency of notation between discrete and continuous
    adjoint problems.
    """
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


def Control(x):
    """
    Dummy `Control` function for consistency of notation between discrete and continuous adjoint
    problems.
    """
    return x


def taylor_test(J, c, dc, dJdm=None):
    """
    Dummy `taylor_test` function for consistency of notation between discrete and continuous
    adjoint problems.
    """
    assert dJdm is not None
    if isinstance(c, Function):
        c = c.dat.data
    elif isinstance(c, list) and isinstance(c[0], Function):
        c = np.array([ci.dat.data[0] for ci in c])
    if isinstance(dc, Function):
        dc = c.dat.data
    elif isinstance(dc, list) and isinstance(dc[0], Function):
        dc = np.array([dci.dat.data[0] for dci in dc])
    opt.taylor_test(J, dJdm, c, delta_m=dc, verbose=True)

"""
**********************************************************************************************
*  NOTE: This file is based on the SciPy project (https://scipy.org) and contains some       *
*        copied code.                                                                        *
**********************************************************************************************
"""
from thetis import ConvergenceError, print_output

import numpy as np

from adapt_utils.misc import prod


__all__ = ["taylor_test", "StagnationError", "GradientConverged"]


# TODO: Taylor test gradient using Frobenius norm as QoI
def taylor_test(function, gradient, m, delta_m=None, verbose=False, ratio_tol=3.95):
    """
    Apply a 'Taylor test' to verify that the provided `gradient` function is a consistent
    approximation to the provided `function` at point `m`. This is done by choosing a random search
    direction and constructing a sequence of finite difference approximations. If the gradient is
    consistent then the associated Taylor remainder will decrease quadratically.

    :arg function: a scalar valued function with a single vector argument.
    :arg gradient: proposed gradient of above function, to be tested.
    :arg m: vector at which to perform the test.
    :arg delta_m: search direction in which to perform the test.
    :kwarg verbose: toggle printing to screen.
    :kwarg ratio_tol: value which must be exceeded for convergence.
    """
    if verbose:
        print_output(24*"=" + "TAYLOR TEST" + 24*"=")
    m = np.array(m).reshape((prod(np.shape(m)), ))
    delta_m = delta_m or np.random.normal(loc=0.0, scale=1.0, size=m.shape)

    # Evaluate the reduced functional and gradient at the specified control value
    Jm = function(m)
    dJdm = gradient(m).reshape(m.shape)

    # Check that the Taylor remainders decrease quadratically
    remainders = np.zeros(3)
    for i in range(3):
        h = pow(0.5, i)
        if verbose:
            print_output("h = {:.4e}".format(h))
        J_step = function(m + h*delta_m)
        remainders[i] = np.abs(J_step - Jm - h*np.dot(dJdm, delta_m))
        if verbose:
            print_output("Taylor remainder = {:.4e}".format(remainders[i]))
        if i > 0:
            ratio = remainders[i-1]/remainders[i]
            try:
                assert ratio > ratio_tol
            except AssertionError:
                msg = "Taylor remainders do not decrease quadratically (ratio {:.4e} < {:.4e})"
                raise ConvergenceError(msg.format(ratio, ratio_tol))
    if verbose:
        print_output(20*"=" + "TAYLOR TEST PASSED!" + 20*"=")


class StagnationError(Exception):
    """Error raised when an optimisation routine stagnates."""


class GradientConverged(Exception):
    """Exception raised when gradient due to an optimisation routine converges."""

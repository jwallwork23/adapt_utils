from thetis import ConvergenceError, print_output
import numpy as np


__all__ = ["taylor_test", "ConvergenceError" "StagnationError", "GradientConverged"]


# TODO: Taylor test gradient using Frobenius norm as QoI
def taylor_test(function, gradient, m, delta_m=None, verbose=False, ratio_tol=1.95):
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
    m = np.array(m).flatten()
    delta_m = delta_m or np.random.normal(loc=0.0, scale=1.0, size=m.shape)
    assert len(m) == len(delta_m)

    # Evaluate the reduced functional and gradient at the specified control value
    Jm = function(m)
    dJdm = gradient(m).flatten()
    assert len(m) == len(dJdm)

    # Check that the Taylor remainders decrease quadratically
    remainders = np.zeros(4)
    epsilons = [0.01*0.5**i for i in range(4)]
    rates = np.zeros(3)
    for i in range(4):
        h = epsilons[i]
        if verbose:
            print_output("h = {:.4e}".format(h))
        J_step = function(m + h*delta_m)
        remainders[i] = np.abs(J_step - Jm - h*np.dot(dJdm, delta_m))
        if i == 0:
            if verbose:
                print_output("remainder = {:.4e}".format(remainders[i]))
        elif i > 0:
            rates[i-1] = np.log(remainders[i]/remainders[i-1])/np.log(epsilons[i]/epsilons[i-1])
            if verbose:
                msg = "remainder = {:.4e}  convergence rate = {:.2f}"
                print_output(msg.format(remainders[i], rates[i-1]))
            try:
                assert rates[i-1] > ratio_tol
            except AssertionError:
                msg = "Taylor remainders do not decrease quadratically (ratio {:.2f} < {:.2f})"
                raise ConvergenceError(msg.format(rates[i-1], ratio_tol))
    if verbose:
        print_output(20*"=" + "TAYLOR TEST PASSED!" + 20*"=")
    return rates


class StagnationError(Exception):
    """Error raised when an optimisation routine stagnates."""


class GradientConverged(Exception):
    """Exception raised when gradient due to an optimisation routine converges."""

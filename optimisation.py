from thetis import ConvergenceError, print_output
from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _prepare_scalar_function, _line_search_wolfe12, \
        _LineSearchError, _status_message, _epsilon

from adapt_utils.misc import prod
from adapt_utils.norms import vecnorm


__all__ = ["minimise_bfgs", "taylor_test", "StagnationError"]


def _minimize_bfgs(fun, x0, args=(), jac=None, hess_inv=None, callback=None, gtol=1e-5, norm=np.Inf,
        epsilon=_epsilon, maxiter=None, disp=False, retall=False, finite_diff_rel_step=None):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    ***
    Copied from scipy/optimize/optimize.py, with the minor modification that
    the Hessian (inverse) approximation can be passed as an argument.
    ***

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    epsilon : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    retall : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    fprime=jac
    eps = epsilon
    return_all = retall

    retall = return_all

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    if not np.isscalar(old_fval):
        try:
            old_fval = old_fval.item()
        except (ValueError, AttributeError):
            raise ValueError("The user-provided "
                             "objective function must "
                             "return a scalar value.")

    k = 0
    N = len(x0)
    # NOTE: Here is the modification
    I = np.eye(N, dtype=int)
    if hess_inv is None:
        Hk = I
    else:
        Hk = hess_inv

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2
    fval = old_fval

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, order=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, myfprime, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, order=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if np.isinf(rhok):  # this is patch for NumPy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])

        fval = old_fval

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = _status_message['nan']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(
        fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev, njev=sf.ngev, status=warnflag,
        success=(warnflag == 0), message=msg, x=xk, nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result


def minimise_bfgs(rf, scale=1.0, **kwargs):
    """
    Minimise a pyadjoint :class:`ReducedFunctional` `rf` using BFGS.

    Code here is a modified version of that found in pyadjoint/optimization/optimization.py,
    allowing not only the optimised control parameters, but also the SciPy :class:`OptimizeResult`
    object. This contains various other useful things, including the Hessian approximation. This is
    stored as :attr:`hess`.
    """
    rf.scale = scale
    if isinstance(rf, ReducedFunctionalNumPy):
        rf_np = rf
    elif isinstance(rf, ReducedFunctional):
        rf_np = ReducedFunctionalNumPy(rf)
    else:
        rf_np = rf

    # Set controls
    m = [p.data() for p in rf_np.controls]
    m_global = rf_np.obj_to_array(m)

    # Get objective functional and derivatives thereof
    J = rf_np.__call__
    dJ = lambda m: rf_np.derivative(m, forget=False, project=kwargs.pop("project", False))

    # Process keyword arguments
    if "disp" not in kwargs:
        kwargs["disp"] = True
    kwargs["jac"] = dJ

    # Call SciPy's optimize
    res = _minimize_bfgs(J, m_global, **kwargs)

    # Return both optimal controls and also SciPy OptimizeResult object
    rf_np.set_controls(np.array(res["x"]))
    m = [p.data() for p in rf_np.controls]
    return m, res


def taylor_test(function, gradient, m, verbose=False, ratio_tol=3.95):
    """
    Apply a 'Taylor test' to verify that the provided `gradient` function is a consistent
    approximation to the provided `function` at point `m`. This is done by choosing a random search
    direction and constructing a sequence of finite difference approximations. If the gradient is
    consistent then the associated Taylor remainder will decrease quadratically.

    :arg function: a scalar valued function with a single vector argument.
    :arg gradient: proposed gradient of above function, to be tested.
    :arg m: vector at which to perform the test.
    :kwarg verbose: toggle printing to screen.
    :kwarg ratio_tol: value which must be exceeded for convergence.
    """
    if verbose:
        print_output(24*"=" + "TAYLOR TEST" + 24*"=")
    m = np.array(m).reshape((prod(np.shape(m)), ))
    delta_m = np.random.normal(loc=0.0, scale=1.0, size=m.shape)

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

import numpy as np
from scipy.optimize import minimize

from pyadjoint.reduced_functional import ReducedFunctional
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy


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
    if "options" not in kwargs:
        kwargs["options"] = {}
    elif "disp" not in kwargs["options"]:
        kwargs["options"]["disp"] = True
    kwargs["jac"] = dJ

    # Call SciPy's optimize
    res = minimize(J, m_global, method='BFGS', **kwargs)

    # Return both optimal controls and also SciPy OptimizeResult object
    rf_np.set_controls(np.array(res["x"]))
    m = [p.data() for p in rf_np.controls]
    return m, res

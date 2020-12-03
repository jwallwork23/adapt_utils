from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adapt_utils.adapt.recovery import recover_hessian
from adapt_utils.plotting import *


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=[True, False])
def interp(request):
    return request.param


def test_hessian_bowl(interp, plot=False):
    r"""
    Check that the recovered Hessian of the quadratic function

  ..math::
        u(x,y) = \frac12(x^2 + y^2)

    is the identity matrix.

    :arg interp: toggle whether or not to interpolate the sensor
    into :math:`\mathbb P1` space before recovering the Hessian.
    """
    n = 100
    mesh = SquareMesh(n, n, 2, 2)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x-1, y-1]))

    # Construct Hessian
    f = 0.5*(x**2 + y**2)
    if interp:
        f = interpolate(f, FunctionSpace(mesh, "CG", 1))
    H = recover_hessian(f, mesh=mesh)

    # Save entry-wise errors in arrays
    H_arr = [[[], []], [[], []]]
    for k, Hk in enumerate(H.dat.data):
        for i in range(2):
            for j in range(2):
                tgt = 1 if i == j else 0
                tol = 0.8 if interp else 1.0e-6
                assert np.isclose(Hk[i, j], tgt, atol=tol), "{:.3f} vs. {:.3f}".format(Hk[i, j], tgt)
                if plot:
                    H_arr[i][j].append(abs(Hk[i, j] - tgt))
    if not plot:
        return

    # Plot errors on scatterplot
    ones = np.ones(len(H_arr[0][0]))
    fig, axes = plt.subplots(figsize=(5, 6))
    for i in range(2):
        for j in range(2):
            axes.scatter((2*i + j + 1)*ones, H_arr[i][j], marker='x')
    axes.set_yscale('log')
    axes.set_xlim([0, 5])
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    axes.set_xlabel("Hessian component")
    if interp:
        axes.set_yticks([1e-15, 1e-10, 1e-5, 1])
    axes.set_ylabel("Discrepancy")
    savefig("hessian_errors_bowl", "outputs/hessian", extensions=["pdf"])


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    # test_hessian_bowl(True, plot=True)
    test_hessian_bowl(False, plot=True)

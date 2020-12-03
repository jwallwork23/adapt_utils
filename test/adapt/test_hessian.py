from firedrake import *

import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.adapt.recovery import recover_hessian
from adapt_utils.plotting import *


def test_hessian_bowl(plot=False):
    r"""
    Check that the recovered Hessian of the quadratic function

  ..math::
        u(x,y) = \frac12(x^2 + y^2)

    is the identity matrix.
    """
    n = 100
    mesh = SquareMesh(n, n, 2, 2)
    x, y = SpatialCoordinate(mesh)
    mesh.coordinates.interpolate(as_vector([x-1, y-1]))

    # Construct Hessian
    bowl = lambda xx, yy: 0.5*(xx**2 + yy**2)
    H = recover_hessian(bowl(x, y), mesh=mesh)

    # Save entry-wise errors in arrays
    H_arr = [[[], []], [[], []]]
    for k, Hk in enumerate(H.dat.data):
        for i in range(2):
            for j in range(2):
                tgt = 1 if i == j else 0
                assert np.isclose(Hk[i, j], tgt, atol=0.001), "{:.3f} vs. {:.3f}".format(Hk[i, j], tgt)
                if plot:
                    H_arr[i][j].append(Hk[i, j] - tgt)
    if not plot:
        return

    # Plot errors on scatterplot
    ones = np.ones(len(H_arr[0][0]))
    fig, axes = plt.subplots(figsize=(5, 6))
    for i in range(2):
        for j in range(2):
            axes.scatter((2*i + j + 1)*ones, H_arr[i][j], marker='x')
    axes.set_xlim([0, 5])
    axes.set_xticks([1, 2, 3, 4])
    axes.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"])
    axes.set_xlabel("Hessian component")
    axes.set_yticks([-3e-7, -2e-7, -1e-7, 0, 1e-7])
    axes.set_ylabel("Discrepancy")
    savefig("hessian_errors_bowl", "outputs/hessian", extensions=["pdf"])


if __name__ == "__main__":
    test_hessian_bowl(plot=True)

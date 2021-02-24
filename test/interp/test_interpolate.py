from firedrake import *
import firedrake.supermeshing as supermesh
import matplotlib.pyplot as plt
from adapt_utils.plotting import *
from adapt_utils.interpolation import *


def setup():
    Hs = UnitSquareMesh(20, 25, diagonal='left')
    Ht = UnitSquareMesh(20, 20, diagonal='right')
    Vs = FunctionSpace(Hs, "CG", 1)
    Vt = FunctionSpace(Ht, "CG", 1)
    xs, ys = SpatialCoordinate(Hs)
    s_init = interpolate(sin(pi*xs)*sin(pi*ys), Vs)
    s = Function(Vs, name="Source").assign(s_init)
    t = Function(Vt, name="Target")
    return s_init, s, t


def plot_error(error, error_type, interp='interpolation'):
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(100*np.array(error))
    axes.set_xlabel("Number of interpolation steps")
    axes.set_ylabel(r"$\mathcal L_2$ error (\%)")
    axes.grid(True)
    savefig("{:s}_error_{:s}".format(error_type, interp), "plots", ["pdf"])


def plot_field(f, fname, levels=np.linspace(-1.0e-06, 1.05, 8), ticks=np.linspace(0, 1.05, 8)):
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(f, axes=axes, levels=levels), ax=axes)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig(fname, "plots", ["jpg"])


# ---------------------------
# standard tests for pytest
# ---------------------------

def test_point_interpolate(plot=False):
    """
    Ping-pong test for hand-coded point interpolation.
    """
    s_init, s, t = setup()
    N = 100
    mass_init = assemble(s_init*dx)
    l2_error = []
    mass_error = []
    s.assign(s_init)
    for i in range(N):
        point_interpolate(s, t)
        point_interpolate(t, s)
        l2_error.append(errornorm(s, s_init)/norm(s_init))
        mass_error.append(abs(assemble(s*dx) - mass_init)/abs(mass_init))
    assert np.isclose(l2_error[-2], l2_error[-1])
    assert np.isclose(mass_error[-2], mass_error[-1])
    if plot:
        plot_error(l2_error, "l2", interp="interpolation")
        plot_error(mass_error, "mass", interp="interpolation")
        plot_field(s_init, "source_for_interpolation")
        plot_field(s, "after_100_interpolations")


def test_supermesh_projection(plot=False):
    """
    Ping-pong test for hand-coded supermesh projection.
    """

    # Setup operators
    s_init, s, t = setup()
    Vs, Vt = s.function_space(), t.function_space()
    M_st = supermesh.assemble_mixed_mass_matrix(Vs, Vt)
    M_ts = supermesh.assemble_mixed_mass_matrix(Vt, Vs)
    M_s = assemble(inner(TrialFunction(Vs), TestFunction(Vs))*dx).M.handle
    M_t = assemble(inner(TrialFunction(Vt), TestFunction(Vt))*dx).M.handle
    solver_s = PETSc.KSP().create()
    solver_s.setOperators(M_s)
    solver_s.setFromOptions()
    solver_t = PETSc.KSP().create()
    solver_t.setOperators(M_t)
    solver_t.setFromOptions()
    M_ts = M_st.copy()
    M_ts.transpose()

    # Ping pong test
    N = 100
    mass_init = assemble(s_init*dx)
    l2_norm_init = norm(s_init)
    l2_error = []
    mass_error = []
    s.assign(s_init)
    for i in range(N):
        supermesh_project(s, t, mixed_mass_matrix=M_st, solver=solver_t)
        supermesh_project(t, s, mixed_mass_matrix=M_ts, solver=solver_s)
        l2_error.append(errornorm(s, s_init)/l2_norm_init)
        mass_error.append(abs(assemble(s*dx) - mass_init)/abs(mass_init))
    assert np.allclose(mass_error, 0.0, atol=1.0e-04)
    if plot:
        plot_error(l2_error, "l2", interp="projection")
        plot_error(mass_error, "mass", interp="projection")
        levels = np.linspace(-0.15, 1.05, 7)
        plot_field(s_init, "source_for_projection", levels, levels)
        plot_field(s, "after_100_projections", levels, levels)


def test_adjoint_supermesh_projection(plot=False):
    """
    Ping-pong test for hand-coded adjoint supermesh projection.
    """
    s_b, t_b = setup()[1:]
    Vt = t_b.function_space()
    Ht = Vt.mesh()
    xt, yt = SpatialCoordinate(Ht)
    t_b_init = interpolate(sin(pi*xt)*sin(pi*yt), Vt)
    t_b.assign(t_b_init)
    adjoint_supermesh_project(t_b, s_b)
    relative_mass_error = 100*abs(assemble(t_b*dx)-assemble(s_b*dx))/assemble(abs(t_b)*dx)
    assert relative_mass_error > 10.0
    if plot:
        levels = np.linspace(-0.15, 1.05, 7)
        plot_field(t_b, "seed", levels, levels)
        plot_field(s_b, "adjoint_projection", levels, levels)


# ---------------------------
# plotting
# ---------------------------

if __name__ == "__main__":
    test_point_interpolate(plot=True)
    test_supermesh_projection(plot=True)
    test_adjoint_supermesh_projection(plot=True)

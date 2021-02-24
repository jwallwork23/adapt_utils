from firedrake import *
import firedrake.supermeshing as supermesh
from adapt_utils.interpolation import adjoint_supermesh_project


plot = False

# Setup two function spaces and a source
Hs = UnitSquareMesh(20, 25, diagonal='left')
Ht = UnitSquareMesh(20, 20, diagonal='right')
Vs = FunctionSpace(Hs, "CG", 1)
Vt = FunctionSpace(Ht, "CG", 1)
xt, yt = SpatialCoordinate(Ht)
t_b_init = Function(Vt, name="Seed")
t_b_init.interpolate(sin(pi*xt)*sin(pi*yt))
t_b = Function(Vt, name="Seed").assign(t_b_init)
s_b = Function(Vs, name="Reverse mode propagation")

adjoint_supermesh_project(t_b, s_b)
relative_mass_error = 100*abs(assemble(t_b*dx)-assemble(s_b*dx))/assemble(abs(t_b)*dx)
assert relative_mass_error > 10.0

# Plot results
if plot:
    import matplotlib.pyplot as plt
    from adapt_utils.plotting import *
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(t_b, axes=axes, levels=np.linspace(-0.15, 1.05, 7)), ax=axes)
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("seed", "plots", ["jpg"])
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(s_b, axes=axes, levels=np.linspace(-0.15, 1.05, 7)), ax=axes)
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("adjoint_projection", "plots", ["jpg"])

from firedrake import *
import firedrake.supermeshing as supermesh
from adapt_utils.interpolation import supermesh_project


plot = False

# Setup two function spaces and a source
Hs = UnitSquareMesh(20, 25, diagonal='left')
Ht = UnitSquareMesh(20, 20, diagonal='right')
Vs = FunctionSpace(Hs, "CG", 1)
Vt = FunctionSpace(Ht, "CG", 1)
xs, ys = SpatialCoordinate(Hs)
s_init = Function(Vs, name="Initial source")
s_init.interpolate(sin(pi*xs)*sin(pi*ys))
s = Function(Vs, name="Source").assign(s_init)
t = Function(Vt, name="Target")

# Get operators
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

# Plot results
if plot:
    import matplotlib.pyplot as plt
    from adapt_utils.plotting import *
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(100*np.array(l2_error))
    axes.set_xlabel("Number of interpolation steps")
    axes.set_ylabel(r"$\mathcal L_2$ error (\%)")
    axes.grid(True)
    savefig("l2_error_projection", "plots", ["pdf"])
    fig, axes = plt.subplots(figsize=(7, 4))
    axes.plot(100*np.array(mass_error))
    axes.set_xlabel("Number of interpolation steps")
    axes.set_ylabel(r"Mass error (\%)")
    axes.grid(True)
    savefig("mass_error_projection", "plots", ["pdf"])
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(s_init, axes=axes, levels=np.linspace(-0.15, 1.05, 7)), ax=axes)
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("source_for_projection", "plots", ["jpg"])
    fig, axes = plt.subplots(figsize=(3.5, 3))
    cbar = fig.colorbar(tricontourf(s, axes=axes, levels=np.linspace(-0.15, 1.05, 7)), ax=axes)
    cbar.ax.tick_params(labelsize=10)
    axes.axis(False)
    savefig("after_100_projections", "plots", ["jpg"])

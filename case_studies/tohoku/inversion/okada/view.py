from thetis import *

import argparse
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.plotting import *
from adapt_utils.misc import ellipse


parser = argparse.ArgumentParser()
parser.add_argument("level")
args = parser.parse_args()

level = int(args.level)
op = TohokuOkadaBasisOptions(level=level, synthetic=False)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):  # TODO: Consider all gauges and account for arrival/dept times
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip']
op.control_parameters['slip'] = np.load('data/opt_progress_discrete_{:d}_ctrl.npy'.format(level))[-1]
op.control_parameters['rake'] = np.zeros(*np.shape(op.control_parameters['rake']))
num_active_controls = len(op.active_controls)
op.end_time = 60*30


# --- Setup tsunami propagation problem

mesh = op.default_mesh
P2_vec = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
TaylorHood = P2_vec*P1
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))

boundary_conditions = {
    100: ['freeslip', 'dirichlet'],
    200: ['freeslip'],
    300: ['freeslip'],
}
dtc = Constant(op.dt)
n = FacetNormal(mesh)

u, eta = TrialFunctions(TaylorHood)
z, zeta = TestFunctions(TaylorHood)
q_ = Function(TaylorHood)
u_, eta_ = q_.split()

a = inner(z, u)*dx + inner(zeta, eta)*dx
L = inner(z, u_)*dx + inner(zeta, eta_)*dx


def G(uv, elev):
    F = g*inner(z, grad(elev))*dx
    F += f*inner(z, as_vector((-uv[1], uv[0])))*dx
    F += -inner(grad(zeta), b*uv)*dx
    for tag in boundary_conditions:
        if "freeslip" not in boundary_conditions[tag]:
            F += inner(zeta*n, b*uv)*ds(tag)
    return F


a += 0.5*dtc*G(u, eta)
L += -0.5*dtc*G(u_, eta_)

q = Function(TaylorHood)
u, eta = q.split()

bcs = []
for tag in boundary_conditions:
    if "dirichlet" in boundary_conditions[tag]:
        bcs.append(DirichletBC(TaylorHood.sub(1), 0, tag))

params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
}

problem = LinearVariationalProblem(a, L, q, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=params)


# --- Setup source model

tape_tag = 0
q0 = Function(TaylorHood)
q0.assign(0.0)
u0, eta0 = q0.split()
op.create_topography(annotate=False)
eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
eta0.dat.name = "Initial surface"

q_init = Function(TaylorHood)
q_init.project(q0)
num_subfaults = len(op.subfaults)

u_init, eta_init = q_init.split()
fig, axes = plt.subplots(figsize=(6, 6))
eta_min = 1.01*eta_init.vector().gather().min()
eta_max = 1.01*eta_init.vector().gather().max()
tc = tricontourf(eta_init, axes=axes, cmap='coolwarm', levels=np.linspace(eta_min, eta_max, 50))
fig.colorbar(tc, ax=axes)
xg, yg = op.gauges["P02"]["coords"]
axes.set_xlim([xg - 0.25e+06, xg + 0.25e+06]);
axes.set_ylim([yg - 0.4e+06, yg + 0.3e+06]);
op.annotate_plot(axes)
axes.axis(False)
plt.show()
savefig("dislocation_{:d}".format(level), "plots", extensions=["jpg"])
exit(0)


def tsunami_propagation(init):
    """
    Run tsunami propagation, given an initial velocity-elevation tuple.
    """
    q_.assign(init)

    for gauge in gauges:
        op.gauges[gauge]['timeseries'] = []
        op.gauges[gauge]['init'] = Constant(eta_.at(op.gauges[gauge]['coords']))
        op.gauges[gauge]['data'] = []

    t = 0.0
    iteration = 0
    J = 0
    weight = Constant(1.0)
    eta_obs = Constant(0.0)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()

        # Time integrate QoI
        weight.assign(0.5 if np.allclose(t, 0.0) or t >= op.end_time - 0.5*op.dt else 1.0)
        u, eta = q.split()
        for gauge in op.gauges:

            # Point evaluation at gauges
            eta_discrete = eta.at(op.gauges[gauge]['coords']) - op.gauges[gauge]['init'].dat.data[0]
            op.gauges[gauge]['timeseries'].append(eta_discrete)

            # Interpolate observations
            obs = float(op.gauges[gauge]['interpolator'](t))
            op.gauges[gauge]['data'].append(obs)
            eta_obs.assign(obs + op.gauges[gauge]['init'])

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            diff = 0.5*I*(eta - eta_obs)**2
            J = J + assemble(weight*dtc*diff*dx)

        # Increment
        q_.assign(q)
        t += op.dt
        iteration += 1

    assert np.allclose(t, op.end_time), print("mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time))
    return J


# --- Get gauge data

gauges = list(op.gauges.keys())
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
P0 = FunctionSpace(mesh, "DG", 0)
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)

times = np.linspace(0, op.end_time, int(op.end_time/op.dt))
for gauge in gauges:
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)


# --- Forward solve

J = tsunami_propagation(q_init)
print("Quantity of interest = {:.4e}".format(J))


# --- Plot gauge timeseries

fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(24, 8), dpi=100)

for i, gauge in enumerate(gauges):
    ax = axes[i//4, i%4]
    ax.plot(times/60, op.gauges[gauge]['timeseries'], label=gauge)
    ax.plot(times/60, op.gauges[gauge]['data'], 'x')
    ax.legend(handlelength=0, handletextpad=0)
    if i >= 4:
        ax.set_xlabel("Time [minutes]")
    if i%4 == 0:
        ax.set_ylabel("Elevation [m]")
    ax.grid(True)
savefig("discrete_timeseries_{:d}".format(level), "plots", extensions=["pdf"])

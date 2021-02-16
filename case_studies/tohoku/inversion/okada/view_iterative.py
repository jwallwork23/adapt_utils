from thetis import *

import argparse
import matplotlib.pyplot as plt

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.plotting import *
from adapt_utils.misc import ellipse


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-alpha")
parser.add_argument("-num_minutes")
args = parser.parse_args()

level = int(args.level)
op = TohokuOkadaBasisOptions(level=level, synthetic=False)
op.end_time = 60*float(args.num_minutes or 30)
alpha = float(args.alpha or 0.0)
reg = not np.isclose(alpha, 0.0)
alpha /= op.nx*op.ny*25.0e+03*20.0e+03
alpha = Constant(alpha)
gauges = list(op.gauges.keys())
for gauge in gauges:
    # if op.gauges[gauge]['arrival_time'] < op.end_time:  # TODO
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip', 'rake']
fname = 'data/opt_progress_discrete_{:d}_{:s}'
if reg:
    fname += '_reg'
try:
    opt_controls = np.load(fname.format(level, 'ctrl') + '.npy')[-1]
    op.control_parameters['slip'] = opt_controls[:190]
    op.control_parameters['rake'] = opt_controls[190:]
except Exception:
    print("Could not find optimised controls. Proceeding with initial guess.")
num_active_controls = len(op.active_controls)
base_dt = 4
op.dt = base_dt*0.5**level


# --- Setup tsunami propagation problem

# Function spaces
mesh = op.default_mesh
P2 = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)

# Fields
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))
bcs = [DirichletBC(P1, 0, 100)]
params = {"snes_type": "ksponly", "ksp_type": "gmres", "pc_type": "sor"}
dtc = Constant(op.dt)
n = FacetNormal(mesh)

# Solution fields, etc.
u = Function(P2)
eta = Function(P1)
u_trial = TrialFunction(P2)
eta_trial = TrialFunction(P1)
z = TestFunction(P2)
zeta = TestFunction(P1)
u_ = Function(P2)
eta_ = Function(P1)

# Velocity solver
lhs_u = inner(z, u_trial)*dx
rhs_u = inner(z, u_)*dx
rhs_u += -dtc*g*inner(z, grad(eta_))*dx
rhs_u += -dtc*f*inner(z, as_vector((-u_[1], u_[0])))*dx
u_prob = LinearVariationalProblem(lhs_u, rhs_u, u, bcs=[])
u_solver = LinearVariationalSolver(u_prob, solver_parameters=params)

# Elevation solver
lhs_eta = inner(zeta, eta_trial)*dx
rhs_eta = inner(zeta, eta_)*dx
rhs_eta += dtc*inner(grad(zeta), b*u_)*dx
eta_prob = LinearVariationalProblem(lhs_eta, rhs_eta, eta, bcs=bcs)
eta_solver = LinearVariationalSolver(eta_prob, solver_parameters=params)

# --- Setup source model

tape_tag = 0
eta0 = Function(P1)
op.create_topography(annotate=False)
eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
eta0.dat.name = "Initial surface"
num_subfaults = len(op.subfaults)

fig, axes = plt.subplots(figsize=(6, 6))
eta_min = 1.01*eta0.vector().gather().min()
eta_max = 1.01*eta0.vector().gather().max()
tc = tricontourf(eta0, axes=axes, cmap='coolwarm', levels=np.linspace(eta_min, eta_max, 50))
fig.colorbar(tc, ax=axes)
xg, yg = op.gauges["P02"]["coords"]
axes.set_xlim([xg - 0.25e+06, xg + 0.25e+06])
axes.set_ylim([yg - 0.4e+06, yg + 0.3e+06])
op.annotate_plot(axes)
axes.axis(False)
fname = "dislocation_{:d}".format(level)
if reg:
    fname += '_reg'
savefig(fname, "plots", extensions=["jpg"])


# --- Get gauge data

radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)


def tsunami_propagation(init):
    """
    Run tsunami propagation, given an initial velocity-elevation tuple.
    """
    u_.assign(0.0)
    eta_.assign(init)
    t = 0.0
    iteration = 0
    if reg:
        J = assemble(alpha*inner(init, init)*dx)
        print("Regularisation term = {:.4e}".format(J))
    else:
        J = 0
    wq = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        op.gauges[gauge]['data'] = [0.0]
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        op.gauges[gauge]['init_smooth'] = assemble(op.gauges[gauge]['indicator']*eta_*dx)
        op.gauges[gauge]['timeseries'] = [op.gauges[gauge]['init']]
        op.gauges[gauge]['timeseries_smooth'] = [0.0]
        op.gauges[gauge]['diff'] = [op.gauges[gauge]['init']]
        op.gauges[gauge]['diff_smooth'] = [0.0]
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(0.5*wq*dtc*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)
    while t < op.end_time:
        if iteration % int(60/op.dt) == 0:
            print("t = {:2.0f} mins".format(t/60))

        # Solve forward equation at current timestep
        u_solver.solve()
        eta_solver.solve()
        u_.assign(u)
        eta_.assign(eta)
        t += op.dt
        iteration += 1

        # Time integrate QoI
        for gauge in op.gauges:
            if t < op.gauges[gauge]['arrival_time']:
                continue
            elif np.isclose(t, op.gauges[gauge]['arrival_time']):
                wq.assign(0.5*0.5*op.dt)
            elif np.isclose(t, op.gauges[gauge]['departure_time']):
                wq.assign(0.5*0.5*op.dt)
            elif t > op.gauges[gauge]['departure_time']:
                continue
            else:
                wq.assign(0.5*1.0*op.dt)

            # Point evaluation at gauges
            eta_discrete = eta.at(op.gauges[gauge]['coords']) - op.gauges[gauge]['init']
            op.gauges[gauge]['timeseries'].append(eta_discrete)

            # Interpolate observations
            if op.gauges[gauge]['init'] is None:
                op.gauges[gauge]['init'] = eta.at(op.gauges[gauge]['coords'])
            obs = float(op.gauges[gauge]['interpolator'](t))
            op.gauges[gauge]['data'].append(obs)
            eta_obs.assign(obs + op.gauges[gauge]['init'])
            op.gauges[gauge]['diff'].append(0.5*(eta_discrete - obs)**2)

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            diff = 0.5*I*(eta - eta_obs)**2
            J = J + assemble(wq*dtc*diff*dx)
            op.gauges[gauge]['timeseries_smooth'].append(assemble(I*eta*dx) - op.gauges[gauge]['init_smooth'])
            op.gauges[gauge]['diff_smooth'].append(assemble(diff*dx))

    assert np.allclose(t, op.end_time), print("mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time))
    return J


# --- Forward solve

J = tsunami_propagation(eta0)
print("Quantity of interest = {:.4e}".format(J))


# --- Plot gauge timeseries

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(24, 8), dpi=100)
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    ax.plot(times/60, op.gauges[gauge]['timeseries'], label=gauge, color='C0')
    ax.plot(times/60, op.gauges[gauge]['timeseries_smooth'], ':', color='C0')
    ax.plot(times/60, op.gauges[gauge]['data'], 'x', color='C1')
    ax.legend(handlelength=0, handletextpad=0)
    if i >= 4:
        ax.set_xlabel("Time [minutes]")
    if i % 4 == 0:
        ax.set_ylabel(r"Elevation [$\mathrm m$]")
    ax.grid(True)
fname = 'discrete_timeseries_both_{:d}'.format(level)
if reg:
    fname += '_reg'
savefig(fname, 'plots', extensions=['pdf'])

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(24, 8), dpi=100)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    ax.plot(times/60, op.gauges[gauge]['diff'], label=gauge)
    ax.plot(times/60, op.gauges[gauge]['diff_smooth'])
    ax.legend(handlelength=0, handletextpad=0)
    if i >= 4:
        ax.set_xlabel("Time [minutes]")
    if i % 4 == 0:
        ax.set_ylabel(r"Squared error [$\mathrm m^2$]")
    ax.grid(True)
fname = 'discrete_timeseries_error_{:d}'.format(level)
if reg:
    fname += '_reg'
savefig(fname, 'plots', extensions=['pdf'])

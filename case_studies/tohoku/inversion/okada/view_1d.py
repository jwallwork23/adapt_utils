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
parser.add_argument("-initial_guess")
args = parser.parse_args()

level = int(args.level)
alpha = float(args.alpha or 0.0)
reg = not np.isclose(alpha, 0.0)
alpha /= 300.0e+03*150.0e+03
alpha = Constant(alpha)
ig = bool(args.initial_guess or False)
control_parameters = {
    'latitude': [37.52],
    'longitude': [143.05],
    'depth': [12581.10],
    'strike': [198.0],
    'length': [300.0e+03],
    'width': [150.0e+03],
    'slip': [29.5],
    'rake': [90.0],
    'dip': [10.0],
}
op = TohokuOkadaBasisOptions(level=level, nx=1, ny=1, control_parameters=control_parameters)
op.end_time = 60*float(args.num_minutes or 120)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip', 'rake']
fname = 'data/opt_progress_discrete_1d_{:d}_{:s}'
if reg:
    fname += '_reg'
loaded = False
try:
    assert not ig
    opt_controls = np.load(fname.format(level, 'ctrl') + '.npy')[-1]
    op.control_parameters['slip'] = [opt_controls[0]]
    op.control_parameters['rake'] = [opt_controls[1]]
    op.control_parameters['dip'] = [opt_controls[2]]
    loaded = True
except Exception:
    print("Could not find optimised controls. Proceeding with initial guess.")
    fname += '_ig'
num_active_controls = len(op.active_controls)


# --- Setup tsunami propagation problem

mesh = op.default_mesh
P2_vec = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)
TaylorHood = P2_vec*P1
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))

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
    return F


a += 0.5*dtc*G(u, eta)
L += -0.5*dtc*G(u_, eta_)

q = Function(TaylorHood)
u, eta = q.split()
params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
}
problem = LinearVariationalProblem(a, L, q, bcs=DirichletBC(TaylorHood.sub(1), 0, 100))
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
# xg, yg = op.gauges["P02"]["coords"]
# axes.set_xlim([xg - 0.25e+06, xg + 0.25e+06])
# axes.set_ylim([yg - 0.4e+06, yg + 0.3e+06])
op.annotate_plot(axes)
axes.axis(False)
fname = "dislocation_1d_{:d}".format(level)
if reg:
    fname += '_reg'
if not loaded:
    fname += '_ig'
savefig(fname, "plots", extensions=["jpg"])


def tsunami_propagation(init):
    """
    Run tsunami propagation, given an initial velocity-elevation tuple.
    """
    q_.assign(init)

    for gauge in gauges:
        op.gauges[gauge]['data'] = [0.0]
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        op.gauges[gauge]['init_smooth'] = assemble(op.gauges[gauge]['indicator']*eta_*dx)
        op.gauges[gauge]['timeseries'] = [op.gauges[gauge]['init']]
        op.gauges[gauge]['timeseries_smooth'] = [0.0]
        op.gauges[gauge]['diff'] = [op.gauges[gauge]['init']]
        op.gauges[gauge]['diff_smooth'] = [0.0]

    t = 0.0
    if reg:
        J = assemble(alpha*inner(init, init)*dx)
        print("Regularisation term = {:.4e}".format(J))
    else:
        J = 0
    weight = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in op.gauges:
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(0.5*weight*dtc*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)
        t += op.dt

        # Time integrate QoI
        weight.assign(0.5 if np.allclose(t, 0.0) or t >= op.end_time - 0.5*op.dt else 1.0)
        for gauge in op.gauges:

            # Point evaluation at gauges
            eta_discrete = eta.at(op.gauges[gauge]['coords']) - op.gauges[gauge]['init']
            op.gauges[gauge]['timeseries'].append(eta_discrete)

            # Interpolate observations
            obs = float(op.gauges[gauge]['interpolator'](t))
            op.gauges[gauge]['data'].append(obs)
            eta_obs.assign(obs + op.gauges[gauge]['init'])
            op.gauges[gauge]['diff'].append(0.5*(eta_discrete - obs)**2)

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            diff = 0.5*I*(eta - eta_obs)**2
            J = J + assemble(weight*dtc*diff*dx)
            op.gauges[gauge]['timeseries_smooth'].append(assemble(I*eta*dx) - op.gauges[gauge]['init_smooth'])
            op.gauges[gauge]['diff_smooth'].append(assemble(diff*dx))

    assert np.allclose(t, op.end_time), print("mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time))
    return J


# --- Get gauge data

radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)


# --- Forward solve

J = tsunami_propagation(q_init)
print("Quantity of interest = {:.4e}".format(J))


# --- Plot gauge timeseries

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(17, 13), dpi=100)
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    ax.plot(times/60, op.gauges[gauge]['timeseries'], label=gauge, color='C0')
    ax.plot(times/60, op.gauges[gauge]['timeseries_smooth'], ':', color='C0')
    ax.plot(times/60, op.gauges[gauge]['data'], 'x', color='C1')
    leg = ax.legend(handlelength=0, handletextpad=0, fontsize=20)
    for item in leg.legendHandles:
        item.set_visible(False)
    if i//len(gauges) == 3:
        ax.set_xlabel("Time [minutes]")
    if i % 4 == 0:
        ax.set_ylabel(r"Elevation [$\mathrm m$]")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    ax.set_xlim([op.gauges[gauge]["arrival_time"]/60, op.gauges[gauge]["departure_time"]/60])
    ax.grid(True)
fname = 'discrete_timeseries_both_1d_{:d}'.format(level)
if reg:
    fname += '_reg'
if not loaded:
    fname += '_ig'
savefig(fname, 'plots', extensions=['pdf'])

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(17, 13), dpi=100)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    ax.plot(times/60, op.gauges[gauge]['diff'], label=gauge)
    ax.plot(times/60, op.gauges[gauge]['diff_smooth'])
    ax.legend(handlelength=0, handletextpad=0, fontsize=20)
    if i//len(gauges) == 3:
        ax.set_xlabel("Time [minutes]")
    if i % 4 == 0:
        ax.set_ylabel(r"Squared error [$\mathrm m^2$]")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    ax.set_xlim([op.gauges[gauge]["arrival_time"]/60, op.gauges[gauge]["departure_time"]/60])
    ax.grid(True)
fname = 'discrete_timeseries_error_1d_{:d}'.format(level)
if reg:
    fname += '_reg'
if not loaded:
    fname += '_ig'
savefig(fname, 'plots', extensions=['pdf'])

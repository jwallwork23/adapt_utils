from thetis import *

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.misc import ellipse
from adapt_utils.plotting import *
from adapt_utils.swe.tsunami.conversion import lonlat_to_utm


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-category")
parser.add_argument("-initial_guess")
parser.add_argument("-uniform_slip")
parser.add_argument("-uniform_rake")
args = parser.parse_args()

level = int(args.level)
category = args.category or 'all'
if 'gps' in category:
    category = 'near_field_gps'
elif 'mid' in category:
    category = 'mid_field_pressure'
elif 'far' in category:
    category = 'far_field_pressure'
elif 'south' in category:
    category = 'southern_pressure'
assert category in (
    'all',
    'near_field_gps',
    'near_field_pressure',
    'mid_field_pressure',
    'far_field_pressure',
    'southern_pressure',
)
ig = bool(args.initial_guess or False)
op = TohokuOkadaBasisOptions(level=level)
op.end_time = 60*120
op.dt = 4*0.5**level
gauges = list(op.gauges.keys())
print(gauges)
fname = 'data/opt_progress_discrete_{:d}_{:s}'.format(level, category) + '_{:s}'
loaded = False
if args.uniform_slip is not None:
    op.control_parameters['slip'] = float(args.uniform_slip)*np.ones(190)
if args.uniform_rake is not None:
    op.control_parameters['rake'] = float(args.uniform_rake)*np.ones(190)
try:
    assert not ig
    print(fname.format('ctrl') + '.npy')
    opt_controls = np.load(fname.format('ctrl') + '.npy')[-1]
    op.control_parameters['slip'] = opt_controls[0::2]
    op.control_parameters['rake'] = opt_controls[1::2]
    if op.control_parameters['rake'].max() > 90.0:
        print("WARNING: Unrealistic rake {:.2f}".format(op.control_parameters['rake'].max()))
    # op.control_parameters['dip'] = opt_controls[3::2]
    # op.control_parameters['strike'] = opt_controls[4::2]
    loaded = True
except Exception:
    print("Could not find optimised controls. Proceeding with initial guess.")
    fname += '_ig'

mesh = op.default_mesh
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)


# --- Setup source model

tape_tag = 0
eta0 = Function(P1)
dislocation = Function(P1)
op.create_topography(annotate=False)
dislocation.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
num_subfaults = len(op.subfaults)
eta0.assign(dislocation)

# --- Plot initial dislocation field

lonlat_corners = [(138, 32), (148, 42), (138, 42)]
utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
xlim = [utm_corners[0][0], utm_corners[1][0]]
ylim = [utm_corners[0][1], utm_corners[2][1]]
figure, axes = plt.subplots(ncols=2, figsize=(15, 6))
axes[0].axis(False)
axes = axes[1]

# Plot over whole domain
# levels = np.linspace(-6, 10, 51)
levels = 51
cbar = figure.colorbar(tricontourf(eta0, levels=levels, axes=axes, cmap='coolwarm'), ax=axes)
cbar.set_label(r"Dislocation $[\mathrm m]$")
# cbar.set_ticks(np.linspace(-5, 10, 4))
op.annotate_plot(axes)
axes.axis(False)

# Add zoom
axins = zoomed_inset_axes(axes, 2.5, bbox_to_anchor=[750, 525])  # zoom-factor: 2.5
tricontourf(eta0, levels=51, axes=axins, cmap='coolwarm')
axins.axis(False)
axins.set_xlim(xlim)
axins.set_ylim(ylim)
mark_inset(axes, axins, loc1=1, loc2=1, fc="none", ec="0.5")

# Save
fname = "dislocation_{:d}_{:s}".format(level, category)
if not loaded:
    fname += '_ig'
savefig(fname, "plots", extensions=["jpg"], tight=False)

# --- Setup tsunami propagation problem

b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
c_sq = g*b
dtc = Constant(op.dt)
dtc_sq = Constant(op.dt**2)
n = FacetNormal(mesh)

eta = TrialFunction(P1)
phi = TestFunction(P1)
eta_ = Function(P1)
eta__ = Function(P1)

a = inner(eta, phi)*dx
L = inner(2*eta_ - eta__, phi)*dx - dtc_sq*inner(c_sq*grad(eta_), grad(phi))*dx
# L += dtc_sq*phi*c_sq*dot(grad(eta_), n)*ds

eta = Function(P1, name="Free surface elevation")
params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
}
prob = LinearVariationalProblem(a, L, eta, bcs=DirichletBC(P1, 0, 100))
solver = LinearVariationalSolver(prob, solver_parameters=params)


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
    eta_.assign(init)
    eta__.assign(init)
    t = op.dt
    J = 0
    wq = Constant(0.5*0.5*op.dt)
    eta_obs = Constant(0.0)

    # Setup QoI
    for gauge in gauges:
        op.gauges[gauge]['init'] = None
        op.gauges[gauge]['init_smooth'] = None
        if t < op.gauges[gauge]['arrival_time']:
            op.gauges[gauge]['timeseries'] = []
            op.gauges[gauge]['timeseries_smooth'] = []
            op.gauges[gauge]['diff'] = []
            op.gauges[gauge]['diff_smooth'] = []
            continue
        op.gauges[gauge]['data'] = [0.0, 0.0]
        op.gauges[gauge]['init'] = eta__.at(op.gauges[gauge]['coords'])
        op.gauges[gauge]['init_smooth'] = assemble(op.gauges[gauge]['indicator']*eta__*dx)
        op.gauges[gauge]['timeseries'] = [0.0, 0.0]
        op.gauges[gauge]['timeseries_smooth'] = [0.0, 0.0]
        op.gauges[gauge]['diff'] = [0.0, 0.0]
        op.gauges[gauge]['diff_smooth'] = [0.0, 0.0]
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta__ - eta_obs)**2*dx)
        J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta_ - eta_obs)**2*dx)

    # Enter timeloop
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        eta__.assign(eta_)
        eta_.assign(eta)
        t += op.dt
        print("t = {:.0f} mins".format(t/60))

        # Time integrate QoI
        for gauge in op.gauges:
            if t < op.gauges[gauge]['arrival_time']:
                continue
            elif np.allclose(t, op.gauges[gauge]['arrival_time']):
                wq.assign(0.5*0.5*op.dt)
            elif np.allclose(t, op.gauges[gauge]['departure_time']):
                wq.assign(0.5*0.5*op.dt)
            elif t > op.gauges[gauge]['departure_time']:
                continue
            else:
                wq.assign(0.5*1.0*op.dt)

            # Point evaluation at gauges
            if op.gauges[gauge]['init'] is None:
                op.gauges[gauge]['init'] = eta.at(op.gauges[gauge]['coords'])
                op.gauges[gauge]['init_smooth'] = assemble(op.gauges[gauge]['indicator']*eta*dx)
            eta_discrete = eta.at(op.gauges[gauge]['coords']) - op.gauges[gauge]['init']
            op.gauges[gauge]['timeseries'].append(eta_discrete)

            # Interpolate observations
            obs = float(op.gauges[gauge]['interpolator'](t))
            op.gauges[gauge]['data'].append(obs)
            eta_obs.assign(obs + op.gauges[gauge]['init'])
            op.gauges[gauge]['diff'].append(0.5*(eta_discrete - obs)**2)

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            diff = I*(eta - eta_obs)**2
            J = J + assemble(wq*diff*dx)
            op.gauges[gauge]['timeseries_smooth'].append(assemble(I*eta*dx) - op.gauges[gauge]['init_smooth'])
            op.gauges[gauge]['diff_smooth'].append(assemble(0.5*diff*dx))

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return J


# --- Forward solve

J = tsunami_propagation(eta0)
print("Quantity of interest = {:.4e}".format(J))


# --- Plot gauge timeseries

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(17, 13), dpi=100)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    Tstart = op.gauges[gauge]['arrival_time']/60
    Tend = op.gauges[gauge]['departure_time']/60
    times = np.linspace(Tstart, Tend, len(op.gauges[gauge]['timeseries']))
    ax.plot(times, op.gauges[gauge]['timeseries'], label=gauge, color='C0')
    ax.plot(times, op.gauges[gauge]['timeseries_smooth'], ':', color='C0')
    ax.plot(times, op.gauges[gauge]['data'], 'x', color='C1', markevery=15*2**level)
    prop = {}
    if op.gauges[gauge]['class'] == category:
        prop['weight'] = 'bold'
    leg = ax.legend(handlelength=0, handletextpad=0, fontsize=20, prop=prop)
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
    ax.set_xlim([Tstart, Tend])
    ax.grid(True)
fname = 'discrete_timeseries_both_{:d}_{:s}'.format(level, category)
if not loaded:
    fname += '_ig'
savefig(fname, 'plots', extensions=['pdf'])

fig, axes = plt.subplots(ncols=4, nrows=len(gauges)//4, figsize=(17, 13), dpi=100)
for i, gauge in enumerate(gauges):
    ax = axes[i//4, i % 4]
    Tstart = op.gauges[gauge]['arrival_time']/60
    Tend = op.gauges[gauge]['departure_time']/60
    times = np.linspace(Tstart, Tend, len(op.gauges[gauge]['timeseries']))
    ax.plot(times, op.gauges[gauge]['diff'], label=gauge)
    ax.plot(times, op.gauges[gauge]['diff_smooth'])
    prop = {}
    if op.gauges[gauge]['class'] == category:
        prop['weight'] = 'bold'
    ax.legend(handlelength=0, handletextpad=0, fontsize=20, prop=prop)
    if i//len(gauges) == 3:
        ax.set_xlabel("Time [minutes]")
    if i % 4 == 0:
        ax.set_ylabel(r"Squared error [$\mathrm m^2$]")
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    ax.set_xlim([Tstart, Tend])
    ax.grid(True)
fname = 'discrete_timeseries_error_{:d}_{:s}'.format(level, category)
if not loaded:
    fname += '_ig'
savefig(fname, 'plots', extensions=['pdf'])

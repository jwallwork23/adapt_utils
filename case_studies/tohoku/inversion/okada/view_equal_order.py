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
parser.add_argument("-num_minutes")
parser.add_argument("-initial_guess")
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
op.end_time = 60*float(args.num_minutes or 120)
gauges = list(op.gauges.keys())
print(gauges)
fname = 'data/opt_progress_discrete_{:d}_{:s}'.format(level, category) + '_{:s}'
loaded = False
try:
    assert not ig
    print(fname.format('ctrl') + '.npy')
    opt_controls = np.load(fname.format('ctrl') + '.npy')[-1]
    op.control_parameters['slip'] = opt_controls[:190]
    op.control_parameters['rake'] = opt_controls[190:380]
    # op.control_parameters['dip'] = opt_controls[380:570]
    # op.control_parameters['strike'] = opt_controls[570:]
    loaded = True
except Exception:
    print("Could not find optimised controls. Proceeding with initial guess.")
    fname += '_ig'

# --- Setup tsunami propagation problem

mesh = op.default_mesh
P1DG_vec = VectorFunctionSpace(mesh, "DG", 1)
P1DG = FunctionSpace(mesh, "DG", 1)
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)
EqualOrder = P1DG_vec*P1DG
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))

dtc = Constant(op.dt)
n = FacetNormal(mesh)

u, eta = TrialFunctions(EqualOrder)
z, zeta = TestFunctions(EqualOrder)
q_ = Function(EqualOrder)
u_, eta_ = q_.split()


def G(uv, elev):
    head_star = avg(elev) + sqrt(b/g)*jump(uv, n)
    hu_star = b*(avg(uv) + sqrt(g/b)*jump(elev, n))
    c = sqrt(g*b)

    F = -g*elev*nabla_div(z)*dx
    F += g*head_star*jump(z, n)*dS
    F += c*dot(uv, n)*dot(z, n)*ds
    F += 0.5*g*elev*dot(z, n)*ds(100)

    F += f*inner(z, as_vector((-uv[1], uv[0])))*dx

    F += -inner(grad(zeta), b*uv)*dx
    F += inner(jump(zeta, n), b*hu_star)*dS
    F += 0.5*zeta*b*dot(uv, n)*ds
    F += zeta*c*elev*ds(100)

    return F


a = inner(z, u)*dx + inner(zeta, eta)*dx + 0.5*dtc*G(u, eta)
L = inner(z, u_)*dx + inner(zeta, eta_)*dx - 0.5*dtc*G(u_, eta_)

q = Function(EqualOrder)
u, eta = q.split()

params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
}
problem = LinearVariationalProblem(a, L, q)
solver = LinearVariationalSolver(problem, solver_parameters=params)


# --- Setup source model

tape_tag = 0
eta0 = Function(P1)
op.create_topography(annotate=False)
eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
eta0.dat.name = "Initial surface"
num_subfaults = len(op.subfaults)

q_init = Function(EqualOrder)
u_init, eta_init = q_init.split()
eta_init.interpolate(eta0)


# --- Plot initial dislocation field

lonlat_corners = [(138, 32), (148, 42), (138, 42)]
utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
xlim = [utm_corners[0][0], utm_corners[1][0]]
ylim = [utm_corners[0][1], utm_corners[2][1]]
figure, axes = plt.subplots(ncols=2, figsize=(15, 6))
axes[0].axis(False)
axes = axes[1]

# Plot over whole domain
levels = np.linspace(-6, 10, 51)
cbar = figure.colorbar(tricontourf(eta0, levels=levels, axes=axes, cmap='coolwarm'), ax=axes)
cbar.set_label(r"Dislocation $[\mathrm m]$")
cbar.set_ticks(np.linspace(-5, 10, 4))
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
    q_.assign(init)

    for gauge in gauges:
        op.gauges[gauge]['data'] = [0.0]
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        op.gauges[gauge]['init_smooth'] = assemble(op.gauges[gauge]['indicator']*eta_*dx)
        op.gauges[gauge]['timeseries'] = [0.0]
        op.gauges[gauge]['timeseries_smooth'] = [0.0]
        op.gauges[gauge]['diff'] = [0.0]
        op.gauges[gauge]['diff_smooth'] = [0.0]

    t = 0.0
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
        print("t = {:.0f} mins".format(t/60))

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
    ax.plot(times/60, op.gauges[gauge]['data'], 'x', color='C1', markevery=2**level)
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
fname = 'discrete_timeseries_both_{:d}_{:s}'.format(level, category)
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
fname = 'discrete_timeseries_error_{:d}_{:s}'.format(level, category)
if not loaded:
    fname += '_ig'
savefig(fname, 'plots', extensions=['pdf'])

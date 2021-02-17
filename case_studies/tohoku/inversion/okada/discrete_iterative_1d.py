"""
Use the same iterative solver approach as `discrete_iterative.py`, but only consider one 300km x 150km
fault, implying just two control parameters. Slip is bounded from below by zero, whereas rake is in the
range [0, 90] degrees.
"""
from thetis import *
from firedrake_adjoint import *

import adolc
import argparse
import scipy.optimize as so
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.misc import ellipse
from adapt_utils.norms import vecnorm
import adapt_utils.optimisation as opt
from adapt_utils.plotting import *


# Parsing
parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-taylor")
parser.add_argument("-gtol")
parser.add_argument("-maxiter")
# parser.add_argument("-alpha")  # TODO
parser.add_argument("-num_minutes")
# parser.add_argument("-resume")  # TODO
parser.add_argument("-plot")
args = parser.parse_args()

# Parameters
level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
maxiter = int(args.maxiter or 1000)
taylor = bool(args.taylor or False)
control_parameters = {
    'latitude': [37.52],
    'longitude': [143.05],
    'depth': [12581.10],
    'strike': [198.0],
    'dip': [10.0],
    'length': [300.0e+03],
    'width': [150.0e+03],
    'slip': [29.5],
    'rake': [90.0],
}
op = TohokuOkadaBasisOptions(level=level, nx=1, ny=1, control_parameters=control_parameters)
op.end_time = 60*float(args.num_minutes or 120)
gauges = list(op.gauges.keys())
for gauge in gauges:
    # if gauge[:2] not in ('P0', '80'):
    # if gauge[:2] != 'P0':
    if op.gauges[gauge]['operator'] != 'NOAA':  # Dart gauges only
        op.gauges.pop(gauge)
    elif op.gauges[gauge]['arrival_time'] >= op.end_time:
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip', 'rake']
num_active_controls = len(op.active_controls)
base_dt = 4
op.dt = base_dt*0.5**level

# I/O
fname = 'data/opt_progress_discrete_1d_{:d}_{:s}'
logname = 'data/discrete_1d_{:d}'.format(level)
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
op._feval = 0

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

# Write to pyadolc's tape
tape_tag = 0
eta0 = Function(P1)
op.create_topography(annotate=True, tag=tape_tag, separate_faults=False)
eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
stats = adolc.tapestats(tape_tag)
for key in stats:
    print("ADOL-C: {:20s}: {:d}".format(key.lower(), stats[key]))
num_subfaults = len(op.subfaults)

if bool(args.plot or False):
    import matplotlib.pyplot as plt
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
    savefig("initial_guess_1d_{:d}".format(level), "plots", extensions=["jpg"])

# Gauge indicators
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)


def solve_forward(init):
    u_.assign(0.0)
    eta_.assign(init)
    t = 0.0
    J = 0
    iteration = 0
    wq = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        op.gauges[gauge]['init'] = None
        if t < op.gauges[gauge]['arrival_time']:
            continue
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)
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

            # Interpolate observations
            if op.gauges[gauge]['init'] is None:
                op.gauges[gauge]['init'] = eta.at(op.gauges[gauge]['coords'])
            eta_obs.assign(float(op.gauges[gauge]['interpolator'](t)) + op.gauges[gauge]['init'])

            # Continuous form of error
            J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return J


# Write to pyadjoint's tape
J = solve_forward(eta0)
print("Quantity of interest = {:.4e}".format(J))
c = Control(eta0)
stop_annotating()
rf_pyadjoint = ReducedFunctional(J, c)

# Taylor test tsunami propagation model
np.random.seed(0)
if taylor:
    m0 = Function(P1).assign(eta0)
    dm0 = Function(P1)
    dm0.dat.data[:] = np.random.rand(*dm0.dat.data.shape)*m0.dat.data
    m0 += dm0
    minconv = taylor_test(rf_pyadjoint, m0, dm0)
    assert minconv > 1.90, minconv


def rf_pyadolc(m, keep=0):
    eta0_tmp = Function(P1)
    eta0_tmp.dat.data[op.indices] = adolc.zos_forward(tape_tag, m, keep=keep)
    return eta0_tmp


def reduced_functional(m):
    """
    Compose both unrolled tapes
    """
    op._J = rf_pyadjoint(rf_pyadolc(m, keep=0))
    return op._J


def gradient(m):
    """
    Apply the chain rule to both tapes
    """
    rf_pyadjoint(rf_pyadolc(m, keep=1))
    dJdeta0 = rf_pyadjoint.derivative()
    dJdeta0 = dJdeta0.dat.data[op.indices]
    return adolc.fos_reverse(tape_tag, dJdeta0)


# Taylor test full source-tsunami model
if taylor:
    m0 = np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
    dm0 = np.random.rand(*m0.shape)
    m0 += dm0
    opt.taylor_test(reduced_functional, gradient, m0, dm0, verbose=True)


def reduced_functional__save(m):
    op._J = reduced_functional(m)
    op._feval += 1
    msg = "slip {:11.4e}  rake {:5.2f}  functional {:15.8e}"
    print(msg.format(m[0], m[1], op._J))
    return op._J


def gradient__save(m):
    dJdm = gradient(m)
    g = vecnorm(dJdm, order=np.Inf)
    msg = "slip {:11.4e}  rake {:5.2f}  functional {:15.8e}  gradient {:15.8e}"
    print(msg.format(m[0], m[1], op._J, g))
    op.control_trajectory.append(m)
    op.functional_trajectory.append(op._J)
    op.gradient_trajectory.append(dJdm)
    np.save(fname.format(level, 'ctrl'), op.control_trajectory)
    np.save(fname.format(level, 'func'), op.functional_trajectory)
    np.save(fname.format(level, 'grad'), op.gradient_trajectory)
    if abs(g) < gtol:
        callback(m)
        raise opt.GradientConverged
    return dJdm


def callback(m):
    print("Line search complete")
    op.line_search_trajectory.append(m)
    np.save(fname.format(level, 'ls'), op.line_search_trajectory)


# Run optimisation
print_output("Optimisation begin...")
c = np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
opt_kwargs = {
    'maxiter': maxiter,
    'pgtol': gtol,
    'fprime': gradient__save,
    'callback': callback,
    'bounds': [(0.0, np.Inf), (0.0, 90.0)],
}
tic = perf_counter()
try:
    out = so.fmin_l_bfgs_b(reduced_functional__save, c, **opt_kwargs)
except opt.GradientConverged:
    out = (
        op.control_trajectory[-1],
        op.functional_trajectory[-1],
        {
            'warnflag': 2,
            'grad': op.gradient_trajectory[-1],
            'funcalls': op._feval,
            'nit': len(op.line_search_trajectory),
        }
    )
cpu_time = perf_counter() - tic
with open(logname + '.log', 'w+') as log:
    log.write("slip minimiser:       {:.8e}\n".format(out[0][0]))
    log.write("rake minimiser:       {:.8e}\n".format(out[0][1]))
    log.write("minimum:              {:.8e}\n".format(out[1]))
    log.write("function evaluations: {:d}\n".format(out[3]['funcalls']))
    log.write("gradient evaluations: {:d}\n".format(len(op.gradient_trajectory)))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

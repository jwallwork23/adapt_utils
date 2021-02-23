from thetis import *
from firedrake_adjoint import *

import adolc
import argparse
import scipy.optimize as so
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
import adapt_utils.optimisation as opt
from adapt_utils.norms import vecnorm
from adapt_utils.misc import ellipse


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("categories")
parser.add_argument("-taylor_test_okada")
parser.add_argument("-taylor_test_tsunami")
parser.add_argument("-taylor_test")
parser.add_argument("-normalise")
parser.add_argument("-load")
parser.add_argument("-gtol")
parser.add_argument("-maxiter")
parser.add_argument("-num_minutes")
parser.add_argument("-uniform_slip")
parser.add_argument("-uniform_rake")
parser.add_argument("-active_controls")
args = parser.parse_args()

level = int(args.level)
gauge_classifications = (
    'all',
    'near_field_gps',
    'near_field_pressure',
    'mid_field_pressure',
    'far_field_pressure',
    'southern_pressure',
)
if 'all' in args.categories:
    categories = 'all'
    gauge_classifications_to_consider = gauge_classifications[1:]
else:
    categories = args.categories.split(',')
    gauge_classifications_to_consider = []
    for category in categories:
        assert category in gauge_classifications
        gauge_classifications_to_consider.append(category)
    categories = '_'.join(categories)
gtol = float(args.gtol or 1.0e-08)
maxiter = int(args.maxiter or 1000)
normalise = bool(args.normalise or False)
fname = 'data/opt_progress_discrete_{:d}_{:s}'.format(level, categories) + '_{:s}'
logname = 'data/discrete_{:d}_{:s}'.format(level, categories)
op = TohokuOkadaBasisOptions(level=level)
op.end_time = 60*float(args.num_minutes or 120)
op.gauge_classifications_to_consider = gauge_classifications_to_consider
op.active_controls = (args.active_controls or 'slip,rake').split(',')
num_active_controls = len(op.active_controls)
if num_active_controls == 0:
    raise ValueError("No active controls set.")
if args.load is not None:
    if args.load not in op.gauge_classifications_to_consider:
        op.gauge_classifications_to_consider.append(args.load)
    fname_ = 'data/opt_progress_discrete_{:d}_{:s}'.format(level, args.load) + '_{:s}'
    print("Loading ", fname_.format('ctrl') + '.npy')
    opt_controls = np.load(fname_.format('ctrl') + '.npy')[-1]
    i = 0
    for control in ('slip', 'rake', 'dip', 'strike'):
        if control in op.active_controls:
            op.control_parameters[control] = opt_controls[i::2]
            i += 1
if args.uniform_slip is not None:
    op.control_parameters['slip'] = float(args.uniform_slip)*np.ones(190)
if args.uniform_rake is not None:
    op.control_parameters['rake'] = float(args.uniform_rake)*np.ones(190)
op.get_gauges()
gauges = list(op.gauges.keys())
latest = 0.0
for gauge in gauges:
    if op.gauges[gauge]['arrival_time'] >= op.end_time:
        op.gauges.pop(gauge)
    latest = max(latest, op.gauges[gauge]['departure_time'])
op.end_time = min(op.end_time, latest)
gauges = list(op.gauges.keys())
print(gauges)
print(op.end_time)

op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
op._feval = 0
mesh = op.default_mesh
P2_vec = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)
TaylorHood = P2_vec*P1

# --- Setup source model

tape_tag = 0
q0 = Function(TaylorHood)
u0, eta0 = q0.split()
dislocation = Function(P1)
with stop_annotating():
    op.create_topography(annotate=True, tag=tape_tag, separate_faults=False)
    dislocation.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
eta0.interpolate(dislocation)

stats = adolc.tapestats(tape_tag)
for key in stats:
    print("ADOL-C: {:20s}: {:d}".format(key.lower(), stats[key]))
num_subfaults = len(op.subfaults)


def okada_source(m, keep=1):
    """
    Compute the dislocation field due to a (flattened) array of
    active control parameters by replaying pyadolc's tape.

    :kwarg keep: toggle whether to flag for a reverse propagation.
    """
    return adolc.zos_forward(tape_tag, m, keep=1)


def gradient_okada(m, m_b=None):
    """
    Compute the gradient of the Okada source model with respect
    to a (flattened) array of active control parameters by
    reverse propagation on pyadolc's tape.
    """
    if m_b is None:
        dislocation = okada_source(m)  # noqa
        m_b = np.ones(len(op.indices))
    return adolc.fos_reverse(tape_tag, m_b)


if bool(args.taylor_test_okada or False):
    """
    Consider the reduced functional

        J(m) = e . S(m)

    Then dJdm is the same as propagating e through the reverse mode of AD on S.
    """
    print("Taylor test Okada...")
    # np.random.seed(0)
    _rf_okada = lambda m: np.sum(okada_source(m))
    _gradient_okada = lambda _: gradient_okada(_, np.ones(len(op.indices)))
    m_init = 0.7*np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
    minconv = opt.taylor_test(_rf_okada, _gradient_okada, m_init, verbose=True)
    assert minconv > 1.90


# --- Setup coupling

def tsunami_ic(dislocation):
    """
    Set the initial velocity-elevation tuple for the tsunami
    propagation model, given some dislocation field.
    """
    init = Function(TaylorHood)
    u_init, eta_init = init.split()
    eta_init.dat.data[op.indices] = dislocation
    return init


def tsunami_ic_inverse(init):
    """
    Extract the dislocation field associated with an initial
    velocity-elevation tuple for the tsunami propagation model.
    """
    u_init, eta_init = init.split()
    return eta_init.dat.data[op.indices]


if bool(args.taylor_test_okada or False):
    print("Taylor test coupling...")
    # np.random.seed(0)
    _rf_coupling = lambda m: np.sum(tsunami_ic(okada_source(m)).dat.data)
    _gradient_coupling = lambda _: gradient_okada(_, np.ones(len(op.indices)))
    m_init = 0.7*np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
    minconv = opt.taylor_test(_rf_coupling, _gradient_coupling, m_init, verbose=True)
    assert minconv > 1.90


# --- Setup tsunami propagation problem

b = Function(P1).assign(op.set_bathymetry(P1))
f = Function(P1).assign(op.set_coriolis(P1))
g = Constant(op.g)
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


# --- Get gauge data

radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
for gauge in gauges:
    loc = op.gauges[gauge]['coords']
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)
    op.gauges[gauge]['weight'] = 1.0
    if normalise:
        t = 0.0
        maxvalue = 0.0
        while t < op.end_time - 1.0e-05:
            if t < op.gauges[gauge]['arrival_time'] or t > op.gauges[gauge]['departure_time']:
                t += op.dt
                continue
            maxvalue = max(maxvalue, op.gauges[gauge]['interpolator'](t)**2)
            t += op.dt
        assert not np.isclose(maxvalue, 0.0)
        op.gauges[gauge]['weight'] /= maxvalue


def tsunami_propagation(init):
    """
    Run tsunami propagation, given an initial velocity-elevation tuple.
    """
    q_.assign(init)
    t = 0.0
    wq = Constant(0.0)
    eta_obs = Constant(0.0)

    # Setup QoI
    J = 0
    for gauge in op.gauges:
        op.gauges[gauge]['init'] = None
        if t < op.gauges[gauge]['arrival_time']:
            continue
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        eta_obs.assign(op.gauges[gauge]['init'])
        wq.assign(0.5*0.5*op.dt*op.gauges[gauge]['weight'])
        J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta_ - eta_obs)**2*dx)

    # Enter timeloop
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)
        t += op.dt

        # Time integrate QoI
        for gauge in op.gauges:
            if t < op.gauges[gauge]['arrival_time']:
                continue
            elif np.allclose(t, op.gauges[gauge]['arrival_time']):
                wq.assign(0.5*0.5*op.dt*op.gauges[gauge]['weight'])
            elif np.allclose(t, op.gauges[gauge]['departure_time']):
                wq.assign(0.5*0.5*op.dt*op.gauges[gauge]['weight'])
            elif t > op.gauges[gauge]['departure_time']:
                continue
            else:
                wq.assign(0.5*1.0*op.dt*op.gauges[gauge]['weight'])

            # Interpolate observations
            if op.gauges[gauge]['init'] is None:
                op.gauges[gauge]['init'] = eta.at(op.gauges[gauge]['coords'])
            eta_obs.assign(float(op.gauges[gauge]['interpolator'](t)) + op.gauges[gauge]['init'])

            # Continuous form of error
            J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return J


# --- Forward solve

c = Control(q0)
J = tsunami_propagation(q0)
stop_annotating()
rf_tsunami = ReducedFunctional(J, c)


# --- Taylor test

if bool(args.taylor_test_tsunami or False):
    """
    J(q0) is the time integrated sum of square differences, averaged over neighbourhoods of each
    gauge. dJdq0 is given by propagating unity through the reverse mode of AD.
    """
    print("Taylor test tsunami...")
    # np.random.seed(0)
    test = Function(q0)
    test.dat.data[:] *= 0.7
    dtest = Function(q0)
    dtest.dat.data[:] = np.random.rand(*dtest.dat.data.shape)*test.dat.data
    minconv = taylor_test(rf_tsunami, test, dtest)
    assert minconv > 1.90


def reduced_functional(m):
    """
    Compose the okada source and tsunami propagation model,
    interfacing with `tsunami_ic`.
    """
    src = okada_source(m)
    init = tsunami_ic(src)
    return rf_tsunami(init)


def gradient(_):
    """
    Compose the gradient functions for the Okada source model
    and the tsunami propagation model, interfacing with the
    inverse of `tsunami_ic`.
    """
    dJdq0 = rf_tsunami.derivative()
    dJdS = tsunami_ic_inverse(dJdq0)
    return gradient_okada(_, m_b=dJdS)


# --- Taylor test

if bool(args.taylor_test or False):
    print("Taylor test coupled...")
    # np.random.seed(0)
    m_init = 0.7*np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
    minconv = opt.taylor_test(reduced_functional, gradient, m_init, verbose=True)
    assert minconv > 1.90


# --- Optimisation

def reduced_functional__save(m):
    """
    Compose both unrolled tapes
    """
    op._J = reduced_functional(m)
    op._feval += 1
    msg = "functional {:15.8e}"
    print(msg.format(op._J))
    return op._J


def gradient__save(m):
    """
    Apply the chain rule to both tapes
    """
    dJdm = gradient(m)
    g = vecnorm(dJdm, order=np.Inf)
    msg = "functional {:15.8e} gradient {:15.8e}"
    print(msg.format(op._J, g))
    op.control_trajectory.append(m)
    op.functional_trajectory.append(op._J)
    op.gradient_trajectory.append(dJdm)
    np.save(fname.format('ctrl'), op.control_trajectory)
    np.save(fname.format('func'), op.functional_trajectory)
    np.save(fname.format('grad'), op.gradient_trajectory)
    return dJdm


def callback(m):
    print("Line search complete")
    op.line_search_trajectory.append(m)
    np.save(fname.format('ls'), op.line_search_trajectory)


print_output("Run optimisation...")
m_init = np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
bound_dict = {"slip": (0, np.Inf), "rake": (0, 90), "dip": (0, 90), "strike": (-np.Inf, np.Inf)}
bounds = [bound_dict[control] for subfault in op.subfaults for control in op.active_controls]
kwargs = dict(
    fprime=gradient__save,
    callback=callback,
    pgtol=gtol,
    maxiter=maxiter,
    bounds=bounds,
)
tic = perf_counter()
x, f, d = so.fmin_l_bfgs_b(reduced_functional__save, m_init, **kwargs)
print(d['task'])
cpu_time = perf_counter() - tic
with open(logname + '.log', 'w+') as log:
    log.write("minimum:              {:.8e}\n".format(op.functional_trajectory[-1]))
    log.write("function evaluations: {:d}\n".format(op._feval))
    log.write("gradient evaluations: {:d}\n".format(len(op.gradient_trajectory)))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

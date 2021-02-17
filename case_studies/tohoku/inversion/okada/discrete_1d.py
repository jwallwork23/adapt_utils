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
parser.add_argument("-gtol")
parser.add_argument("-maxiter")
parser.add_argument("-alpha")
parser.add_argument("-num_minutes")
parser.add_argument("-resume")
args = parser.parse_args()

level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
maxiter = int(args.maxiter or 1000)
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
alpha = float(args.alpha or 0.0)
reg = not np.isclose(alpha, 0.0)
alpha /= control_parameters['length'][0]*control_parameters['width'][0]
alpha = Constant(alpha)
fname = 'data/opt_progress_discrete_1d_{:d}_{:s}'
logname = 'data/discrete_1d_{:d}'.format(level)
if reg:
    fname += '_reg'
    logname += '_reg'
resume = bool(args.resume or False)
op = TohokuOkadaBasisOptions(nx=1, ny=1, level=level, control_parameters=control_parameters)
op.end_time = 60*float(args.num_minutes or 60)
if resume:
    op.control_trajectory = list(np.load(fname.format(level, 'ctrl') + '.npy'))
    op.functional_trajectory = list(np.load(fname.format(level, 'func') + '.npy'))
    op.gradient_trajectory = list(np.load(fname.format(level, 'grad') + '.npy'))
    op.line_search_trajectory = list(np.load(fname.format(level, 'ls') + '.npy'))
    op.control_parameters['slip'] = op.control_trajectory[-1][0]
    op.control_parameters['rake'] = op.control_trajectory[-1][1]
    op.control_parameters['dip'] = op.control_trajectory[-1][2]
    with open(logname + '.log', 'r') as log:
        log.readline()
        op._feval = int(log.readline().split(':')[1])
else:
    op.control_trajectory = []
    op.functional_trajectory = []
    op.gradient_trajectory = []
    op.line_search_trajectory = []
    op._feval = 0
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge not in ('KPG1', 'KPG2', '21418'):
        op.gauges.pop(gauge)
    elif op.gauges[gauge]['arrival_time'] >= op.end_time:
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip', 'rake', 'dip']
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
with stop_annotating():
    q0.assign(0.0)
    u0, eta0 = q0.split()
    op.create_topography(annotate=True, tag=tape_tag, separate_faults=False)
    eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)

q_init = Function(TaylorHood)
q_init.project(q0)

stats = adolc.tapestats(tape_tag)
for key in stats:
    print("ADOL-C: {:20s}: {:d}".format(key.lower(), stats[key]))
num_subfaults = len(op.subfaults)


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
    t = 0.0
    wq = Constant(0.5*0.5*op.dt)
    eta_obs = Constant(0.0)

    # Setup QoI
    J = 0 if not reg else assemble(alpha*inner(init, init)*dx)
    for gauge in op.gauges:
        op.gauges[gauge]['init'] = None
        if t < op.gauges[gauge]['arrival_time']:
            continue
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(wq*op.gauges[gauge]['indicator']*(eta - eta_obs)**2*dx)

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
                wq.assign(0.5*0.5*op.dt)
            elif np.allclose(t, op.gauges[gauge]['departure_time']):
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


# --- Forward solve

J = tsunami_propagation(q_init)
if resume:
    J_ = op.functional_trajectory[-1]
    # assert np.isclose(J, J_), "Expected {:.4e}, got {:.4e}.".format(J_, J)
    print("Expected {:.4e}, got {:.4e}.".format(J_, J))


c = Control(q_init)
stop_annotating()
rf_tsunami = ReducedFunctional(J, c)


def okada_source(m, keep=1):
    """
    Compute the dislocation field due to a (flattened) array of
    active control parameters by replaying pyadolc's tape.

    :kwarg keep: toggle whether to flag for a reverse propagation.
    """
    return adolc.zos_forward(tape_tag, m, keep=1)


def gradient_tsunami(q0):
    """
    Compute the gradient of the tsunami propagation model with
    respect to an initial velocity-elevation tuple by reverse
    propagation on pyadjoint's tape.
    """
    J = rf_tsunami(q0)  # noqa
    return rf_tsunami.derivative()


def gradient_okada(m, m_b=None):
    """
    Compute the gradient of the Okada source model with respect
    to a (flattened) array of active control parameters by
    reverse propagation on pyadolc's tape.
    """
    if m_b is None:
        dislocation = okada_source(m)  # noqa
        m_b = np.one(len(op.indices))
    return adolc.fos_reverse(tape_tag, m_b)


def tsunami_ic(dislocation):
    """
    Set the initial velocity-elevation tuple for the tsunami
    propagation model, given some dislocation field.
    """
    q0 = Function(TaylorHood)
    u0, eta0 = q0.split()
    eta0.dat.data[op.indices] = dislocation
    return q0


def reduced_functional(m):
    """
    Compose the okada source and tsunami propagation model,
    interfacing with `tsunami_ic`.
    """
    q0 = tsunami_ic(okada_source(m))
    return rf_tsunami(q0)
    # return tsunami_propagation(q0)


def tsunami_ic_inverse(q0):
    """
    Extract the dislocation field associated with an initial
    velocity-elevation tuple for the tsunami propagation model.
    """
    u0, eta0 = q0.split()
    return eta0.dat.data[op.indices]


def gradient(m):
    """
    Compose the gradient functions for the Okada source model
    and the tsunami propagation model, interfacing with the
    inverse of `tsunami_ic`.
    """
    q0 = tsunami_ic(okada_source(m))
    dJdq0 = gradient_tsunami(q0)
    dJdeta0 = tsunami_ic_inverse(dJdq0)
    return gradient_okada(m, m_b=dJdeta0)


# --- Taylor tests  # FIXME

# np.random.seed(0)
m_init = np.concatenate([op.control_parameters[ctrl] for ctrl in op.active_controls])
# m_init += 0.2
# # opt.taylor_test(okada_source, gradient_okada, m_init, verbose=True)  # TODO: TESTME
# opt.taylor_test(reduced_functional, gradient, m_init, verbose=True)


# --- Optimisation

def reduced_functional__save(m):
    """
    Compose both unrolled tapes
    """
    op._J = reduced_functional(m)
    op._feval += 1
    msg = "slip {:5.2f}  rake {:5.2f}  dip {:5.2f}  functional {:15.8e}"
    print(msg.format(m[0], m[1], m[2], op._J))
    return op._J


def gradient__save(m):
    """
    Apply the chain rule to both tapes
    """
    dJdm = gradient(m)
    g = vecnorm(dJdm, order=np.Inf)
    msg = "slip {:5.2f}  rake {:5.2f}  dip {:5.2f}  functional {:15.8e}  gradient {:15.8e}"
    print(msg.format(m[0], m[1], m[2], op._J, g))
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


print_output("Run optimisation...")
kwargs = dict(
    fprime=gradient__save,
    callback=callback,
    pgtol=gtol,
    maxiter=maxiter,
    bounds=[(0.0, np.Inf), (0.0, 90.0), (0.0, 90.0)],
)
tic = perf_counter()
try:
    so.fmin_l_bfgs_b(reduced_functional__save, m_init, **kwargs)
except opt.GradientConverged:
    pass
cpu_time = perf_counter() - tic
with open(logname + '.log', 'w+') as log:
    log.write("slip minimiser:       {:.8e}\n".format(op.control_trajectory[-1][0]))
    log.write("rake minimiser:       {:.4f}\n".format(op.control_trajectory[-1][1]))
    log.write("dip minimiser:        {:.4f}\n".format(op.control_trajectory[-1][2]))
    log.write("minimum:              {:.8e}\n".format(op.functional_trajectory[-1]))
    log.write("function evaluations: {:d}\n".format(op._feval))
    log.write("gradient evaluations: {:d}\n".format(len(op.gradient_trajectory)))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

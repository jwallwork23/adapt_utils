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
args = parser.parse_args()

level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
maxiter = int(args.maxiter or 10000)
op = TohokuOkadaBasisOptions(level=level, synthetic=False)
op.end_time = 60*float(args.num_minutes or 30)
alpha = float(args.alpha or 0.0)/(op.nx*op.ny*25.0e+03*20.0e+03)
reg = not np.isclose(alpha, 0.0)
alpha = Constant(alpha)
gauges = list(op.gauges.keys())
for gauge in gauges:
    # if op.gauges[gauge]['arrival_time'] < op.end_time:  # TODO
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip', 'rake']
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


def tsunami_propagation(init):
    """
    Run tsunami propagation, given an initial velocity-elevation tuple.
    """
    q_.assign(init)
    for gauge in gauges:
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
    t = 0.0
    wq = Constant(0.5*0.5*op.dt)
    eta_obs = Constant(0.0)

    # Setup QoI
    J = 0 if not reg else assemble(alpha*inner(init, init)*dx)
    for gauge in op.gauges:
        if t < op.gauges[gauge]['arrival_time']:
            continue
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
            eta_obs.assign(float(op.gauges[gauge]['interpolator'](t)) + op.gauges[gauge]['init'])

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            J = J + assemble(wq*I*(eta - eta_obs)**2*dx)

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
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

print_output("Run optimisation...")
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
op._feval = 0
fname = 'data/opt_progress_discrete_{:d}_{:s}'
if reg:
    fname += '_reg'


def reduced_functional__save(m):
    """
    Compose both unrolled tapes
    """
    op._J = reduced_functional(m)
    op._feval += 1
    print("functional {:15.8e}".format(op._J))
    return op._J


def gradient__save(m):
    """
    Apply the chain rule to both tapes
    """
    dJdm = gradient(m)
    g = vecnorm(dJdm, order=np.Inf)
    print("functional {:15.8e}  gradient {:15.8e}".format(op._J, g))
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


kwargs = dict(fprime=gradient__save, callback=callback, gtol=gtol, full_output=True, maxiter=maxiter)
tic = perf_counter()
try:
    out = so.fmin_bfgs(reduced_functional__save, m_init, **kwargs)
except opt.GradientConverged:
    out = (op.control_trajectory[-1], op.functional_trajectory[-1], op.gradient_trajectory[-1], op._feval, len(op.gradient_trajectory))
cpu_time = perf_counter() - tic
fname = 'data/discrete_{:d}'.format(level)
if reg:
    fname += '_reg'
with open(fname + '.log', 'w+') as log:
    log.write("minimum:              {:.8e}\n".format(out[1]))
    log.write("function evaluations: {:d}\n".format(out[4]))
    log.write("gradient evaluations: {:d}\n".format(out[5]))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

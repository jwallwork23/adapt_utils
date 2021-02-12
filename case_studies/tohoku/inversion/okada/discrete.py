from thetis import *
from firedrake_adjoint import *

import adolc
import argparse
import scipy.optimize as so
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.optimisation import GradientConverged
from adapt_utils.norms import vecnorm
from adapt_utils.misc import ellipse


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-gtol")
args = parser.parse_args()

level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
op = TohokuOkadaBasisOptions(level=level, synthetic=False)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):  # TODO: Consider all gauges and account for arrival/dept times
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
op.active_controls = ['slip']
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
with stop_annotating():
    q0.assign(0.0)
    u0, eta0 = q0.split()
    op.create_topography(annotate=True, tag=tape_tag, separate_faults=False)
    eta0.dat.data[op.indices] = op.fault.dtopo.dZ.reshape(op.fault.dtopo.X.shape)
eta0.dat.name = "Initial surface"

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
        op.gauges[gauge]['timeseries'] = []
        op.gauges[gauge]['diff'] = []
        op.gauges[gauge]['timeseries_smooth'] = []
        op.gauges[gauge]['diff_smooth'] = []
        op.gauges[gauge]['init'] = None
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
            eta_discrete = eta.at(op.gauges[gauge]["coords"])
            if op.gauges[gauge]['init'] is None:
                op.gauges[gauge]['init'] = eta_discrete
            eta_discrete -= op.gauges[gauge]['init']
            op.gauges[gauge]['timeseries'].append(eta_discrete)

            # Interpolate observations
            obs = float(op.gauges[gauge]['interpolator'](t))
            eta_obs.assign(obs)
            op.gauges[gauge]['data'].append(obs)

            # Discrete form of error
            diff = 0.5*(eta_discrete - eta_obs.dat.data[0])**2
            op.gauges[gauge]['diff'].append(diff)

            # Continuous form of error
            I = op.gauges[gauge]['indicator']
            diff = 0.5*I*(eta - eta_obs)**2
            J += assemble(weight*dtc*diff*dx)
            op.gauges[gauge]['diff_smooth'].append(assemble(diff*dx, annotate=False))
            op.gauges[gauge]['timeseries_smooth'].append(assemble(I*eta_obs*dx, annotate=False))

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
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,), ], mesh), P0)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/assemble(op.gauges[gauge]['indicator']*dx))

times = np.linspace(0, op.end_time, int(op.end_time/op.dt))
for gauge in gauges:
    op.sample_timeseries(gauge, sample=op.gauges[gauge]['sample'], detide=True)


# --- Forward solve

J = tsunami_propagation(q_init)
print("Quantity of interest = {:.4e}".format(J))


c = Control(q_init)
stop_annotating()
rf_pyadjoint = ReducedFunctional(J, c)


def rf_pyadolc(m):
    """
    Unroll PyADOL-C's tape and flag for a reverse propagation.
    """
    m_array = m.reshape((num_active_controls, num_subfaults))
    for i, control in enumerate(op.active_controls):
        assert len(op.control_parameters[control]) == len(m_array[i])
        op.control_parameters[control][:] = m_array[i]
    q0_tmp = Function(TaylorHood)
    q0_tmp.assign(0.0)
    u0_tmp, eta0_tmp = q0_tmp.split()
    eta0_tmp.dat.data[op.indices] = adolc.zos_forward(tape_tag, op.input_vector, keep=1)
    return q0_tmp


# Run optimisation
print_output("Run optimisation...")
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
op._feval = 0


def reduced_functional(m):
    """
    Compose both unrolled tapes
    """
    op._J = rf_pyadjoint(rf_pyadolc(m))
    op._feval += 1
    print("functional {:15.8e}".format(op._J))
    return op._J


def gradient(m):
    """
    Apply the chain rule to both tapes
    """
    # op._J = rf_pyadjoint(rf_pyadolc(m))  # TODO: Check not necessary
    dJdq0 = rf_pyadjoint.derivative()
    dJdu0, dJdeta0 = dJdq0.split()
    dJdeta0 = dJdeta0.dat.data[op.indices]
    dJdm = adolc.fos_reverse(tape_tag, dJdeta0)
    print("functional {:15.8e}  gradient {:15.8e}".format(op._J, vecnorm(dJdm, order=np.Inf)))
    op.control_trajectory.append(m)
    op.functional_trajectory.append(op._J)
    op.gradient_trajectory.append(dJdm)
    np.save('data/opt_progress_continuous_{:d}_ctrl'.format(level), op.control_trajectory)
    np.save('data/opt_progress_continuous_{:d}_func'.format(level), op.functional_trajectory)
    np.save('data/opt_progress_continuous_{:d}_grad'.format(level), op.gradient_trajectory)
    if abs(g) < gtol:
        callback(m)
        raise GradientConverged
    return dJdm


def callback(m):
    print("Line search complete")
    op.line_search_trajectory.append(m)
    np.save('data/opt_progress_continuous_{:d}_ls'.format(level), op.line_search_trajectory)


c = np.array(op.control_parameters['slip']).flatten()
kwargs = dict(fprime=gradient, callback=callback, gtol=gtol, full_output=True, maxiter=1000)
tic = perf_counter()
try:
    out = so.fmin_bfgs(reduced_functional, c, **opt_kwargs)
except GradientConverged:
    out = (op.control_trajectory[-1], op.functional_trajectory[-1], op.gradient_trajectory[-1], op._feval, len(op.gradient_trajectory))
cpu_time = perf_counter() - tic
with open('data/continuous_{:d}.log'.format(level), 'w+') as log:
    log.write("minimum:              {:.8e}\n".format(out[1]))
    log.write("function evaluations: {:d}\n".format(out[4]))
    log.write("gradient evaluations: {:d}\n".format(out[5]))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

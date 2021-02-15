from thetis import *
from firedrake_adjoint import *

import argparse
import scipy.interpolate as si
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse
from adapt_utils.optimisation import GradientConverged


parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-gtol")
args = parser.parse_args()

level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
op = TohokuInversionOptions(level=level)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
op.end_time = 60*30

mesh = op.default_mesh
P2_vec = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
TaylorHood = P2_vec*P1


# --- Setup forward problem

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


# --- Define basis

R = FunctionSpace(mesh, "R", 0)
optimum = 5.0
m = Function(R).assign(optimum)
basis_function = Function(TaylorHood)
psi, phi = basis_function.split()
loc = (0.7e+06, 4.2e+06)
radii = (48e+03, 96e+03)
angle = pi/12
phi.interpolate(gaussian([loc + radii, ], mesh, rotation=angle))


def solve_forward(control, store=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)

    if store:
        for gauge in gauges:
            op.gauges[gauge]['data'] = [eta_.at(op.gauges[gauge]['coords'])]

    t = 0.0
    iteration = 0
    J = 0
    wq = Constant(1.0)
    eta_obs = Constant(0.0)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()

        # Time integrate QoI
        wq.assign(0.5 if np.allclose(t, 0.0) or t >= op.end_time - 0.5*op.dt else 1.0)
        for gauge in op.gauges:

            if store:
                # Point evaluation at gauges
                eta_discrete = eta.at(op.gauges[gauge]['coords'])
                op.gauges[gauge]['data'].append(eta_discrete)
            else:
                # Continuous form of error
                eta_obs.assign(op.gauges[gauge]['data'][iteration])
                J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)

        # Increment
        q_.assign(q)
        t += op.dt
        iteration += 1

    assert np.allclose(t, op.end_time), print("mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time))
    return None if store else J


# --- Gauge indicators

gauges = list(op.gauges.keys())
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
P0 = FunctionSpace(mesh, "DG", 0)
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)


# --- Get 'data'

print("Solve forward to get 'data'...")
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
with stop_annotating():
    solve_forward(m, store=True)
    for gauge in gauges:
        op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['data'])


# --- Annotate tape

print("Solve forward to annotate tape...")
m.assign(10.0)
J = solve_forward(m)
c = Control(m)
stop_annotating()
Jhat = ReducedFunctional(J, c)


# --- Taylor test

print("Taylor test at m = 10...")
dm0 = Function(R).assign(0.1)
minconv = taylor_test(Jhat, m, dm0)
assert minconv > 1.90, minconv


# --- Optimisation

print("Run optimisation...")
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []


def cb_post(j, dj, mm):
    op.control_trajectory.append(mm.dat.data[0])
    op.functional_trajectory.append(j)
    op.gradient_trajectory.append(dj.dat.data[0])
    print("control {:12.8f} functional {:15.8e} gradient {:15.8e}".format(mm.dat.data[0], j, dj.dat.data[0]))
    np.save('data/opt_progress_discrete_{:d}_ctrl'.format(level), op.control_trajectory)
    np.save('data/opt_progress_discrete_{:d}_func'.format(level), op.functional_trajectory)
    np.save('data/opt_progress_discrete_{:d}_grad'.format(level), op.gradient_trajectory)
    if abs(dj.dat.data[0]) < gtol:
        op.line_search_trajectory.append(mm.dat.data[0])
        np.save('data/opt_progress_discrete_{:d}_ls'.format(level), op.line_search_trajectory)
        raise GradientConverged


def cb(mm):
    print("Line search complete")
    op.line_search_trajectory.append(mm[0])
    np.save('data/opt_progress_discrete_{:d}_ls'.format(level), op.line_search_trajectory)


Jhat = ReducedFunctional(J, c, derivative_cb_post=cb_post)
c.assign(10.0)
tic = perf_counter()
try:
    m_opt = minimize(Jhat, method='BFGS', callback=cb, options={'gtol': gtol})
except GradientConverged:
    m_opt = op.control_trajectory[-1]
cpu_time = perf_counter() - tic
with open('data/discrete_{:d}.log'.format(level), 'w+') as log:
    log.write("minimiser:            {:.8e}\n".format(op.control_trajectory[-1]))
    log.write("minimum:              {:.8e}\n".format(op.functional_trajectory[-1]))
    log.write("gradient at min:      {:.8e}\n".format(op.gradient_trajectory[-1]))
    # TODO: function evaluations
    log.write("gradient evaluations: {:d}\n".format(len(op.gradient_trajectory)))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

from thetis import *

import argparse
import scipy.interpolate as si
import scipy.optimize as so
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse
from adapt_utils.optimisation import GradientConverged


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-gtol")
parser.add_argument("-family")
args = parser.parse_args()

# Set parameters
level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
family = args.family or 'dg-cg'
assert family in ('dg-cg', 'cg-cg')
op = TohokuInversionOptions(level=level)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
op.end_time = 60*30

# Create function spaces
mesh = op.default_mesh
P1 = FunctionSpace(mesh, "CG", 1)
if family == 'dg-cg':
    V = VectorFunctionSpace(mesh, "DG", 1)*FunctionSpace(mesh, "CG", 2)
elif family == 'cg-cg':
    V = VectorFunctionSpace(mesh, "CG", 2)*P1

# Setup forward problem
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))
c = g*b
dtc = Constant(op.dt)
n = FacetNormal(mesh)
u, eta = TrialFunctions(V)
z, zeta = TestFunctions(V)
q_ = Function(V)
u_, eta_ = q_.split()


def G(uv, elev):
    F = f*inner(z, as_vector([-uv[1], uv[0]]))*dx
    if family == 'dg-cg':
        F += g*inner(z, grad(elev))*dx
        F += c*dot(uv, n)*dot(z, n)*ds
        F += -0.5*g*elev*dot(z, n)*ds(100)
        F += -inner(grad(zeta), b*uv)*dx
        F += 0.5*zeta*b*dot(uv, n)*ds
        F += zeta*c*elev*ds(100)
    elif family == 'cg-cg':
        F += g*inner(z, grad(elev))*dx
        F += -inner(grad(zeta), b*uv)*dx
    return F


a = inner(z, u)*dx + inner(zeta, eta)*dx + 0.5*dtc*G(u, eta)
L = inner(z, u_)*dx + inner(zeta, eta_)*dx - 0.5*dtc*G(u_, eta_)
q = Function(V)
u, eta = q.split()
params = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
}
bcs = None if 'dg' in family else DirichletBC(V.sub(1), 0, 100)
problem = LinearVariationalProblem(a, L, q, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=params)

# Define basis
R = FunctionSpace(mesh, "R", 0)
optimum = 5.0
m = Function(R).assign(optimum)
basis_function = Function(V)
psi, phi = basis_function.split()
loc = (0.7e+06, 4.2e+06)
radii = (48e+03, 96e+03)
angle = pi/12
phi.interpolate(gaussian([loc + radii], mesh, rotation=angle))

# Define gauge indicators
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
P0 = FunctionSpace(mesh, "DG", 0)
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)


def solve_forward(control, store=False, keep=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)
    for gauge in gauges:
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        if store:
            op.gauges[gauge]['data'] = [op.gauges[gauge]['init']]
    if keep:
        op.eta_saved = [eta_.copy(deepcopy=True)]

    t = 0.0
    iteration = 0
    J = 0
    wq = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        J += assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta_ - eta_obs)**2*dx)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)
        t += op.dt
        iteration += 1

        # Time integrate QoI
        wq.assign(0.5 if t >= op.end_time - 0.5*op.dt else 1.0)
        for gauge in gauges:
            if store:
                # Point evaluation at gauges
                op.gauges[gauge]['data'].append(eta.at(op.gauges[gauge]['coords']))
            else:
                # Continuous form of error
                eta_obs.assign(op.gauges[gauge]['data'][iteration] + op.gauges[gauge]['init'])
                J += assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)
        if keep:
            op.eta_saved.append(eta.copy(deepcopy=True))

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return None if store else J


# Solve forward to get 'data'
print("Solve forward to get 'data'...")
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
solve_forward(m, store=True)
for gauge in gauges:
    op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['data'])

# Setup continuous adjoint
u_star, eta_star = TrialFunctions(V)
q_star_ = Function(V)
u_star_, eta_star_ = q_star_.split()
rhs = Function(P1)


def G_star(uv_star, elev_star):
    F = f*inner(z, as_vector((-uv_star[1], uv_star[0])))*dx
    if family == 'dg-cg':
        F += b*inner(z, grad(elev_star))*dx
        F += c*dot(uv_star, n)*dot(z, n)*ds
        F += -0.5*b*elev_star*dot(z, n)*ds(100)
        F += -g*inner(grad(zeta), uv_star)*dx
        F += 0.5*g*zeta*inner(uv_star, n)*ds
        F += c*inner(zeta*n, uv_star)*ds(100)
    elif family == 'cg-cg':
        F += b*inner(z, grad(elev_star))*dx
        F += -g*inner(grad(zeta), uv_star)*dx
        F += g*inner(zeta*n, uv_star)*ds(100)
    return F


a_star = inner(z, u_star)*dx + inner(zeta, eta_star)*dx + 0.5*dtc*G_star(u_star, eta_star)
L_star = inner(z, u_star_)*dx + inner(zeta, eta_star_)*dx - 0.5*dtc*G_star(u_star_, eta_star_)
L_star += dtc*zeta*rhs*dx
q_star = Function(V)
u_star, eta_star = q_star.split()
adj_problem = LinearVariationalProblem(a_star, L_star, q_star)
adj_solver = LinearVariationalSolver(adj_problem, solver_parameters=params)
for gauge in gauges:
    op.gauges[gauge]['obs'] = Constant(0.0)


def compute_gradient_continuous(control):
    """
    Compute gradient by solving continuous adjoint problem.
    """
    iteration = -1
    t = op.end_time
    while t > 0.0:

        # Evaluate function appearing in RHS
        eta_saved = op.eta_saved[iteration]
        for gauge in gauges:
            op.gauges[gauge]['obs'].assign(op.gauges[gauge]['data'][iteration])
        rhs.interpolate(sum(op.gauges[g]['indicator']*(eta_saved - op.gauges[g]['obs']) for g in gauges))

        # Solve adjoint equation at current timestep
        adj_solver.solve()
        q_star_.assign(q_star)
        t -= op.dt
        iteration -= 1

    assert np.allclose(t, 0.0)
    return assemble(phi*eta_star*dx)


def continuous_rf(control):
    """
    Reduced functional.
    """
    tmp = Function(R).assign(control[0])
    op._J = solve_forward(tmp, keep=True)
    op._feval += 1
    print("control {:12.8f} functional {:15.8e}".format(control[0], op._J))
    return op._J


def continuous_gradient(control):
    tmp = Function(R).assign(control[0])
    g = compute_gradient_continuous(tmp)
    print("control {:12.8f} gradient   {:15.8e}".format(control[0], g))
    op.control_trajectory.append(control[0])
    op.functional_trajectory.append(op._J)
    op.gradient_trajectory.append(g)
    np.save('data/opt_progress_continuous_{:d}_ctrl'.format(level), op.control_trajectory)
    np.save('data/opt_progress_continuous_{:d}_func'.format(level), op.functional_trajectory)
    np.save('data/opt_progress_continuous_{:d}_grad'.format(level), op.gradient_trajectory)
    if abs(g) < gtol:
        cb(control)
        raise GradientConverged
    return g


def cb(control):
    print("Line search complete")
    op.line_search_trajectory.append(control[0])
    np.save('data/opt_progress_continuous_{:d}_ls'.format(level), op.line_search_trajectory)


# Run optimisation
print("Run optimisation...")
m.assign(10.0)
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
op._feval = 0
kwargs = dict(fprime=continuous_gradient, callback=cb, gtol=gtol, full_output=True)
tic = perf_counter()
try:
    out = list(so.fmin_bfgs(continuous_rf, m.dat.data[0], **kwargs))
except GradientConverged:
    out = ([op.control_trajectory[-1]], op.functional_trajectory[-1], [op.gradient_trajectory[-1]], op._feval, len(op.gradient_trajectory))
cpu_time = perf_counter() - tic
with open('data/continuous_{:d}.log'.format(level), 'w+') as log:
    log.write("minimiser:            {:.8e}\n".format(out[0][0]))
    log.write("minimum:              {:.8e}\n".format(out[1]))
    log.write("gradient at min:      {:.8e}\n".format(out[2][0]))
    log.write("function evaluations: {:d}\n".format(out[4]))
    log.write("gradient evaluations: {:d}\n".format(out[5]))
    log.write("CPU time:             {:.2f}\n".format(cpu_time))

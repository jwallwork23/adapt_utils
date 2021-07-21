from thetis import *

import argparse
import scipy.optimize as so
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse
from adapt_utils.optimisation import GradientConverged


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-gtol")
args = parser.parse_args()

# Set parameters
level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
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
V = VectorFunctionSpace(mesh, "CG", 2)*P1

# Setup forward problem
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1)  # .assign(op.set_coriolis(P1))
dtc = Constant(op.dt)
u, eta = TrialFunctions(V)
z, zeta = TestFunctions(V)
q_ = Function(V)
u_, eta_ = q_.split()


def G(uv, elev):
    """
    **HARD-CODED** formulation for LSWE.
    """
    return f*inner(z, perp(uv))*dx \
        + g*inner(z, grad(elev))*dx \
        - inner(grad(zeta), b*uv)*dx


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
bcs = DirichletBC(V.sub(1), 0, 100)
problem = LinearVariationalProblem(a, L, q, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=params)

# Define basis
R = FunctionSpace(mesh, "R", 0)
optimum = 5.0
m = Function(R).assign(optimum)
loc = (0.7e+06, 4.2e+06)
radii = (48e+03, 96e+03)
angle = pi/12
phi = interpolate(gaussian([loc + radii], mesh, rotation=angle), P1)

# Define gauge indicators
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
P0 = FunctionSpace(mesh, "DG", 0)
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)
wq = Constant(1.0)

def solve_forward(control, store=False, keep=False):
    """
    Solve forward problem.
    """
    u_.assign(0.0)
    eta_.project(control*phi)
    if store:
        for gauge in gauges:
            # op.gauges[gauge]['data'] = [eta_.at(op.gauges[gauge]['coords'])]
            op.gauges[gauge]['data'] = []
    if keep:
        # op.eta_saved = [eta_.copy(deepcopy=True)]
        op.eta_saved = []

    t = 0.0
    iteration = 0
    # wq.assign(0.5)
    wq.assign(1.0)
    J = 0
    eta_obs = Constant(0.0)
    # if not store:
    #     for gauge in gauges:
    #         eta_obs.assign(op.gauges[gauge]['data'][iteration])
    #         J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta_ - eta_obs)**2*dx)
    while t < op.end_time - 0.5*op.dt:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)

        # Time integrate QoI
        # wq.assign(0.5 if t >= op.end_time - 0.5*op.dt else 1.0)
        wq.assign(1.0)
        for gauge in gauges:
            if store:
                # Point evaluation at gauges
                op.gauges[gauge]['data'].append(eta.at(op.gauges[gauge]['coords']))
            else:
                # Continuous form of error
                eta_obs.assign(op.gauges[gauge]['data'][iteration])
                J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)
        if keep:
            op.eta_saved.append(eta.copy(deepcopy=True))
        t += op.dt
        iteration += 1

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return None if store else J


# Get 'data'
print("Solve forward to get 'data'...")
solve_forward(m, store=True)

# Setup continuous adjoint
u_star, eta_star = TrialFunctions(V)
q_star_ = Function(V)
u_star_, eta_star_ = q_star_.split()
rhs = Function(P1)


def G_star(uv_star, elev_star):
    """
    **HARD-CODED** formulation for continuous
    adjoint LSWE solved using Taylor-Hood.
    """
    return f*inner(uv_star, perp(z))*dx \
        + g*inner(uv_star, grad(zeta))*dx \
        - inner(grad(elev_star), b*z)*dx


a_star = inner(z, u_star)*dx + inner(zeta, eta_star)*dx + 0.5*dtc*G_star(u_star, eta_star)
L_star = inner(z, u_star_)*dx + inner(zeta, eta_star_)*dx - 0.5*dtc*G_star(u_star_, eta_star_)

eta_saved = Function(P1)
for g in gauges:
    op.gauges[g]['obs'] = Constant(0.0)
    L_star += wq*dtc*zeta*op.gauges[g]['indicator']*(eta_saved - op.gauges[g]['obs'])*dx

q_star = Function(V)
u_star, eta_star = q_star.split()
adj_problem = LinearVariationalProblem(a_star, L_star, q_star, bcs=bcs)
adj_solver = LinearVariationalSolver(adj_problem, solver_parameters=params)


def compute_gradient_continuous(control):
    """
    Compute gradient by solving continuous adjoint problem.
    """
    iteration = len(op.eta_saved)
    for gauge in gauges:
        if iteration != len(op.gauges[gauge]['data']):
            raise Exception("{:d} vs. {:d}".format(iteration, len(op.gauges[gauge]['data'])))
    t = op.end_time
    # t = op.end_time + op.dt
    while t > 0.5*op.dt:
        # wq.assign(0.5 if iteration in (1, len(op.eta_saved)) else 1.0)
        wq.assign(1.0)
        iteration -= 1
        t -= op.dt

        # Update functions appearing in RHS
        eta_saved.assign(op.eta_saved[iteration])
        for gauge in gauges:
            op.gauges[gauge]['obs'].assign(op.gauges[gauge]['data'][iteration])

        # Solve adjoint equation at current timestep
        adj_solver.solve()
        q_star_.assign(q_star)

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

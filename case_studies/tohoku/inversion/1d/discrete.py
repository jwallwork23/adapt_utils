from thetis import *
from firedrake_adjoint import *

import argparse
import scipy.interpolate as si
from time import perf_counter

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse
from adapt_utils.optimisation import GradientConverged


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("level")
parser.add_argument("-gtol")
parser.add_argument("-family")
parser.add_argument("-taylor_test")
args = parser.parse_args()

# Set parameters
level = int(args.level)
gtol = float(args.gtol or 1.0e-08)
family = args.family or 'cg-cg'
assert family in ('dg-dg', 'dg-cg', 'cg-cg')
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
if family == 'dg-dg':
    V = VectorFunctionSpace(mesh, "DG", 1)*FunctionSpace(mesh, "DG", 1)
elif family == 'dg-cg':
    V = VectorFunctionSpace(mesh, "DG", 1)*FunctionSpace(mesh, "CG", 2)
elif family == 'cg-cg':
    V = VectorFunctionSpace(mesh, "CG", 2)*P1

# Setup forward problem
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))
c = sqrt(g*b)
dtc = Constant(op.dt)
n = FacetNormal(mesh)
u, eta = TrialFunctions(V)
z, zeta = TestFunctions(V)
q_ = Function(V)
u_, eta_ = q_.split()


def G(uv, elev):
    """
    **HARD-CODED** formulation for LSWE.

    Uses the same flux terms as Thetis.
    """

    # Coriolis
    F = f*inner(z, as_vector([-uv[1], uv[0]]))*dx

    # Gravity
    if 'cg' in family:
        F += g*inner(z, grad(elev))*dx
        if family == 'dg-cg':
            F += c*dot(uv, n)*dot(z, n)*ds
            F += -0.5*g*elev*dot(z, n)*ds(100)
    else:
        head_star = avg(elev) + sqrt(b/g)*jump(uv, n)
        F = -g*elev*nabla_div(z)*dx
        F += g*head_star*jump(z, n)*dS
        F += c*dot(uv, n)*dot(z, n)*ds
        F += 0.5*g*elev*dot(z, n)*ds(100)

    # HUDiv
    if 'dg' in family:
        F += -inner(grad(zeta), b*uv)*dx
        F += 0.5*zeta*b*dot(uv, n)*ds
        F += zeta*c*elev*ds(100)
        if family == 'dg-dg':
            hu_star = b*(avg(uv) + sqrt(g/b)*jump(elev, n))
            inner(jump(zeta, n), b*hu_star)*dS
    else:
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


def solve_forward(control, store=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)
    for gauge in gauges:
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        if store:
            op.gauges[gauge]['data'] = [op.gauges[gauge]['init']]

    t = 0.0
    iteration = 0
    J = 0
    wq = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta_ - eta_obs)**2*dx)
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
                J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)
    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return None if store else J


# Get 'data'
print("Solve forward to get 'data'...")
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
with stop_annotating():
    solve_forward(m, store=True)
    for gauge in gauges:
        op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['data'])

# Annotate tape
print("Solve forward to annotate tape...")
m.assign(10.0)
J = solve_forward(m)
c = Control(m)
stop_annotating()
Jhat = ReducedFunctional(J, c)

# Taylor test
if bool(args.taylor_test or False):
    print("Taylor test at m = 10...")
    dm0 = Function(R).assign(0.1)
    minconv = taylor_test(Jhat, m, dm0)
    assert minconv > 1.90, minconv


def cb_post(j, dj, control):
    op.control_trajectory.append(control.dat.data[0])
    op.functional_trajectory.append(j)
    op.gradient_trajectory.append(dj.dat.data[0])
    msg = "control {:12.8f} functional {:15.8e} gradient {:15.8e}"
    print(msg.format(control.dat.data[0], j, dj.dat.data[0]))
    np.save('data/opt_progress_discrete_{:d}_ctrl'.format(level), op.control_trajectory)
    np.save('data/opt_progress_discrete_{:d}_func'.format(level), op.functional_trajectory)
    np.save('data/opt_progress_discrete_{:d}_grad'.format(level), op.gradient_trajectory)
    if abs(dj.dat.data[0]) < gtol:
        op.line_search_trajectory.append(control.dat.data[0])
        np.save('data/opt_progress_discrete_{:d}_ls'.format(level), op.line_search_trajectory)
        raise GradientConverged


def cb(control):
    print("Line search complete")
    op.line_search_trajectory.append(control[0])
    np.save('data/opt_progress_discrete_{:d}_ls'.format(level), op.line_search_trajectory)


# Run optimisation
print("Run optimisation...")
op.control_trajectory = []
op.functional_trajectory = []
op.gradient_trajectory = []
op.line_search_trajectory = []
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

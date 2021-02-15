from thetis import *

import argparse
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse


parser = argparse.ArgumentParser()
parser.add_argument("level")
args = parser.parse_args()

level = int(args.level)
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
P0 = FunctionSpace(mesh, "DG", 0)
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
phi.interpolate(gaussian([loc + radii], mesh, rotation=angle))


def solve_forward(control, store=False, keep=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)
    for gauge in gauges:
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]['coords'])
        if store:
            op.gauges[gauge]['data'] = [op.gauges[gauge]['init']]
        op.gauges[gauge]['adjoint_free'] = 0.0
    if keep:
        op.eta_saved = [eta_.copy(deepcopy=True)]

    t = 0.0
    iteration = 0
    J = 0
    weight = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        eta_obs.assign(op.gauges[gauge]['init'])
        I = op.gauges[gauge]['indicator']
        diff = eta_ - eta_obs
        J += assemble(0.5*I*weight*dtc*diff*diff*dx)
        op.gauges[gauge]['adjoint_free'] += assemble(I*weight*dtc*diff*eta_*dx)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)
        t += op.dt
        iteration += 1

        # Time integrate QoI
        weight.assign(0.5 if t >= op.end_time - 0.5*op.dt else 1.0)
        for gauge in gauges:
            if store:
                # Point evaluation at gauges
                op.gauges[gauge]['data'].append(eta.at(op.gauges[gauge]['coords']))
            else:
                # Continuous form of error
                eta_obs.assign(op.gauges[gauge]['data'][iteration] + op.gauges[gauge]['init'])
                I = op.gauges[gauge]['indicator']
                diff = eta - eta_obs
                J += assemble(0.5*I*weight*dtc*diff*diff*dx)
                op.gauges[gauge]['adjoint_free'] += assemble(I*weight*dtc*diff*eta*dx)
        if keep:
            op.eta_saved.append(eta.copy(deepcopy=True))

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return None if store else J


# --- Gauge indicators

radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
for gauge in gauges:
    loc = op.gauges[gauge]["coords"]
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,)], mesh), P0)
    area = assemble(op.gauges[gauge]['indicator']*dx)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/area)


# --- Solve forward to get 'data'

print("Solve forward to get 'data'...")
times = np.linspace(0, op.end_time, int(op.end_time/op.dt)+1)
solve_forward(m, store=True)
for gauge in gauges:
    op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['data'])


# --- Setup continuous adjoint

u_star, eta_star = TrialFunctions(TaylorHood)
q_star_ = Function(TaylorHood)
u_star_, eta_star_ = q_star_.split()

a_star = inner(z, u_star)*dx
L_star = inner(z, u_star_)*dx
a_star += inner(zeta, eta_star)*dx
L_star += inner(zeta, eta_star_)*dx


def G_star(uv_star, elev_star):
    F = -b*inner(z, grad(elev_star))*dx
    F += -f*inner(z, as_vector((-uv_star[1], uv_star[0])))*dx
    F += g*inner(grad(zeta), uv_star)*dx
    for tag in boundary_conditions:
        if "dirichlet" in boundary_conditions[tag]:
            F += -inner(zeta*n, uv_star)*ds(tag)
    return F


a_star += 0.5*dtc*G_star(u_star, eta_star)
L_star += -0.5*dtc*G_star(u_star_, eta_star_)

rhs = Function(P1)
L_star += dtc*zeta*rhs*dx

q_star = Function(TaylorHood)
u_star, eta_star = q_star.split()

adj_bcs = []
for tag in boundary_conditions:
    if "freeslip" not in boundary_conditions[tag]:
        adj_bcs.append(DirichletBC(TaylorHood.sub(1), 0, tag))

adj_problem = LinearVariationalProblem(a_star, L_star, q_star, bcs=adj_bcs)
adj_solver = LinearVariationalSolver(adj_problem, solver_parameters=params)

for gauge in gauges:
    op.gauges[gauge]['obs'] = Constant(0.0)


def compute_gradient_continuous(control):
    """
    Compute gradient by solving continuous adjoint problem.
    """
    solve_forward(control, keep=True)
    iteration = int(op.end_time/op.dt)
    t = op.end_time
    while t > 0.0:

        # Evaluate function appearing in RHS
        eta_saved = op.eta_saved[iteration]
        for gauge in gauges:
            op.gauges[gauge]['obs'].assign(op.gauges[gauge]['data'][iteration-1])
        rhs.interpolate(sum(op.gauges[g]['indicator']*(eta_saved - op.gauges[g]['obs']) for g in gauges))

        # Solve adjoint equation at current timestep
        adj_solver.solve()

        # Increment
        q_star_.assign(q_star)
        t -= op.dt
        iteration -= 1

    assert np.allclose(t, 0.0)
    return assemble(phi*eta_star*dx)


m.assign(10.0)
g_continuous = compute_gradient_continuous(m)
print("Gradient computed using continuous adjoint at   m={:2.0f}: {:.8f}".format(m.dat.data[0], g_continuous))
g_adjoint_free = sum([op.gauges[gauge]['adjoint_free'] for gauge in gauges])/m.dat.data[0]
print("Gradient computed using adjoint free formula at m={:2.0f}: {:.8f}".format(m.dat.data[0], g_adjoint_free))

m.assign(5.0)
g_continuous = compute_gradient_continuous(m)
print("Gradient computed using continuous adjoint at   m={:2.0f}: {:.8f}".format(m.dat.data[0], g_continuous))
g_adjoint_free = sum([op.gauges[gauge]['adjoint_free'] for gauge in gauges])/m.dat.data[0]
print("Gradient computed using adjoint free formula at m={:2.0f}: {:.8f}".format(m.dat.data[0], g_adjoint_free))

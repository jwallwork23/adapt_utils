from thetis import *
from firedrake_adjoint import *

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


def solve_forward(control, store=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)

    for gauge in gauges:
        op.gauges[gauge]['init'] = eta_.at(op.gauges[gauge]["coords"])
        if store:
            op.gauges[gauge]['data'] = [op.gauges[gauge]['init']]

    t = 0.0
    iteration = 0
    J = 0
    wq = Constant(0.5)
    eta_obs = Constant(0.0)
    for gauge in gauges:
        eta_obs.assign(op.gauges[gauge]['init'])
        J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)
    while t < op.end_time:

        # Solve forward equation at current timestep
        solver.solve()
        q_.assign(q)
        t += op.dt
        iteration += 1

        # Time integrate QoI
        wq.assign(0.5 if t >= op.end_time - 0.5*op.dt else 1.0)
        for gauge in op.gauges:

            if store:
                # Point evaluation at gauges
                op.gauges[gauge]['data'].append(eta.at(op.gauges[gauge]['coords']))
            else:
                # Continuous form of error
                eta_obs.assign(op.gauges[gauge]['data'][iteration] + op.gauges[gauge]['init'])
                J = J + assemble(0.5*op.gauges[gauge]['indicator']*wq*dtc*(eta - eta_obs)**2*dx)

    assert np.allclose(t, op.end_time), "mismatching end time ({:.2f} vs {:.2f})".format(t, op.end_time)
    return None if store else J


# --- Gauge indicators

gauges = list(op.gauges.keys())
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
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
g_discrete = Jhat.derivative()
print("Gradient computed using discrete adjoint at m={:2.0f}: {:.8f}".format(m.dat.data[0], g_discrete.dat.data[0]))

m.assign(5.0)
Jhat(m)
g_discrete = Jhat.derivative()
print("Gradient computed using discrete adjoint at m={:2.0f}: {:.8f}".format(m.dat.data[0], g_discrete.dat.data[0]))

from thetis import *

import argparse
import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("level")
args = parser.parse_args()

# Set parameters
level = int(args.level)
op = TohokuInversionOptions(level=level)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
op.end_time = 60*30

# Create function spaces
mesh = op.default_mesh
P2_vec = VectorFunctionSpace(mesh, "CG", 2)
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)
V = P2_vec*P1

# Setup forward problem
b = Function(P1).assign(op.set_bathymetry(P1))
g = Constant(op.g)
f = Function(P1).assign(op.set_coriolis(P1))
dtc = Constant(op.dt)
n = FacetNormal(mesh)
u, eta = TrialFunctions(V)
z, zeta = TestFunctions(V)
q_ = Function(V)
u_, eta_ = q_.split()


def G(uv, elev):
    F = g*inner(z, grad(elev))*dx
    F += f*inner(z, as_vector((-uv[1], uv[0])))*dx
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
problem = LinearVariationalProblem(a, L, q, bcs=DirichletBC(V.sub(1), 0, 100))
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

# Gauge indicators
radius = 20.0e+03*pow(0.5, level)  # The finer the mesh, the more precise the indicator region
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
solve_forward(m, store=True)
for gauge in gauges:
    op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['data'])

# Sample at three points and deduce the minimum
control_parameters = [5, 7.5, 10]
functional_values = []
for c in control_parameters:
    m.assign(c)
    functional_values.append(solve_forward(m))
print("Controls: ", control_parameters)
print("QoIs:     ", functional_values)
l = si.lagrange(control_parameters, functional_values)
dl = l.deriv()
print("Exact gradient at 10.0: {:.4f}".format(dl(10.0)))
print("Exact gradient at  5.0: {:.4f}".format(dl(5.0)))
l_min = -dl.coefficients[1]/dl.coefficients[0]
print("Minimiser of quadratic: {:.4f}".format(l_min))

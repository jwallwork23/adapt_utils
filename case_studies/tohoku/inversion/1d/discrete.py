from thetis import *
from firedrake_adjoint import *

import scipy.interpolate as si

from adapt_utils.case_studies.tohoku.options.options import TohokuInversionOptions
from adapt_utils.misc import gaussian, ellipse


# level = 0
level = 1
op = TohokuInversionOptions(level=level)
gauges = list(op.gauges.keys())
for gauge in gauges:
    if gauge[:2] not in ('P0', '80'):
        op.gauges.pop(gauge)
gauges = list(op.gauges.keys())
print(gauges)
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


def solve_forward(control, store=False, keep=False):
    """
    Solve forward problem.
    """
    q_.project(control*basis_function)

    for gauge in gauges:
        op.gauges[gauge]['timeseries'] = []
        op.gauges[gauge]['diff'] = []
        op.gauges[gauge]['timeseries_smooth'] = []
        op.gauges[gauge]['diff_smooth'] = []
        op.gauges[gauge]['init'] = None
        if store:
            op.gauges[gauge]['data'] = []
        op.gauges[gauge]['adjoint_free'] = 0.0
    if keep:
        u_, eta_ = q_.split()
        op.eta_saved = [eta_.copy(deepcopy=True)]

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
            if store:
                op.gauges[gauge]['data'].append(eta_discrete)
            else:
                eta_obs.assign(op.gauges[gauge]['data'][iteration])

                # Discrete form of error
                diff = eta_discrete - eta_obs.dat.data[0]
                op.gauges[gauge]['diff'].append(diff)

                # Continuous form of error
                I = op.gauges[gauge]['indicator']
                diff = eta - eta_obs
                J += assemble(0.5*I*weight*dtc*diff*diff*dx)
                op.gauges[gauge]['adjoint_free'] += assemble(I*weight*dtc*diff*eta*dx, annotate=False)
                op.gauges[gauge]['diff_smooth'].append(assemble(diff*dx, annotate=False))
                op.gauges[gauge]['timeseries_smooth'].append(assemble(I*eta_obs*dx, annotate=False))

        if keep:
            op.eta_saved.append(eta.copy(deepcopy=True))

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
    op.gauges[gauge]['indicator'] = interpolate(ellipse([loc + (radius,), ], mesh), P0)
    op.gauges[gauge]['indicator'].assign(op.gauges[gauge]['indicator']/assemble(op.gauges[gauge]['indicator']*dx))


# --- Get 'data'

print("Solve forward to get 'data'...")
times = np.linspace(0, op.end_time, int(op.end_time/op.dt))
with stop_annotating():
    solve_forward(m, store=True)
    for gauge in gauges:
        op.gauges[gauge]['interpolator'] = si.interp1d(times, op.gauges[gauge]['timeseries'])


# --- Annotate tape

print("Solve forward to annotate tape...")
m.assign(10.0)
J = solve_forward(m)
print("Quantity of interest = {:.4e}".format(J))
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


Jhat = ReducedFunctional(J, c, derivative_cb_post=cb_post)


def cb(mm):
    op.line_search_trajectory.append(mm[0])
    print("Line search complete")


c.assign(10.0)
m_opt = minimize(Jhat, method='BFGS', callback=cb, options={'gtol': 1.0e-08})


# --- Only store data from successful line searches

i = 0
indices = [0]
for j, ctrl in enumerate(op.control_trajectory):
    if i == len(op.line_search_trajectory):
        break
    if np.isclose(ctrl, op.line_search_trajectory[i]):
        indices.append(j)
        i += 1
op.control_trajectory = [op.control_trajectory[i] for i in indices]
op.functional_trajectory = [op.functional_trajectory[i] for i in indices]
op.gradient_trajectory = [op.gradient_trajectory[i] for i in indices]
np.save('data/opt_progress_discrete_{:d}_ctrl'.format(level), op.control_trajectory)
np.save('data/opt_progress_discrete_{:d}_func'.format(level), op.functional_trajectory)
np.save('data/opt_progress_discrete_{:d}_grad'.format(level), op.gradient_trajectory)


# --- Taylor test at 'optimum'

print("Taylor test at 'optimum'...")
m0.assign(m_opt)
dm0.assign(0.1)
minconv = taylor_test(Jhat, m0, dm0)
assert minconv > 1.90, minconv

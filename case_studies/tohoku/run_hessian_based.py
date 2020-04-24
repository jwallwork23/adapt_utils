from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem
from adapt_utils.adapt.metric import steady_metric

import argparse


parser = argparse.ArgumentParser(prog="run_hessian_based")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-norm_order", help="p for Lp normalisaton")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-target", help="Target space-time complexity")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()


# --- Setup

# Set parameters for fixed mesh run
op = TohokuOptions(
    level=int(args.level or 0),
    approach='fixed_mesh',
    adapt_field=args.adapt_field or 'elevation',
    plot_pvd=True,
    debug=bool(args.debug or False),
    norm_order=int(args.norm_order or 1),  # TODO: L-inf normalisation
    target=float(args.target or 1.0e+06),
)
op.end_time = float(args.end_time or op.end_time)
print(op.dt)

# Setup solver
swp = TsunamiProblem(op, levels=0)
swp.setup_solver_forward()

average_hessian = Function(swp.P1_ten, name="Average Hessian")
average_hessians = []
timestep_integrals = []
hessian_file = File(os.path.join(swp.di, 'hessian.pvd'))


# --- Callbacks


def hessian(sol):
    uv, elev = sol.split()
    if op.adapt_field == 'elevation':
        return steady_metric(elev, noscale=True, op=op)
    else:
        raise NotImplementedError  # TODO


timestep = lambda sol: 1.0/op.dt


def extra_setup():

    # Number of timesteps per export (trivial for constant dt)
    swp.callbacks["timestep"] = callback.TimeIntegralCallback(
        timestep, swp.solver_obj, swp.solver_obj.timestepper, name="timestep", append_to_log=False)
    swp.solver_obj.add_callback(swp.callbacks["timestep"], 'timestep')

    # Time integrated Hessian over each window
    swp.callbacks["average_hessian"] = callback.TimeIntegralCallback(
        hessian, swp.solver_obj, swp.solver_obj.timestepper, name="average_hessian", append_to_log=False)
    swp.solver_obj.add_callback(swp.callbacks["average_hessian"], 'timestep')


swp.extra_setup = extra_setup


# --- Exports


def get_export_func(solver_obj):

    def export_func():
        if solver_obj.iteration == 0:
            return

        # Extract time averaged Hessian and reset to zero for next window
        average_hessian.interpolate(swp.callbacks["average_hessian"].get_value())
        hessian_file.write(average_hessian)
        average_hessians.append(average_hessian.copy(deepcopy=True))
        swp.callbacks["average_hessian"].integrant = 0

        # Extract timesteps per export and reset to zero for next window
        timestep_integrals.append(swp.callbacks["timestep"].get_value())
        swp.callbacks["timestep"].integrant = 0.0

    return export_func


op.get_export_func = get_export_func


# --- Run fixed mesh
swp.solve()
print_output("Quantity of interest: {:.4e}".format(swp.callbacks["qoi"].get_value()))


# --- Global normalisation coefficient

p = op.norm_order
d = swp.mesh.topological_dimension()
glob_norm = 0.0
for i, H in enumerate(average_hessians):
    Ki = assemble(pow(det(H), p/(2*p + d))*dx)
    glob_norm += Ki*pow(timestep_integrals[i], 2*p/(2*p + d))
glob_norm = pow(glob_norm, -2/d)
print_output("Global normalisation factor: {:.4e}".format(glob_norm))


# --- Construct Hessians on each window

metric_file = File(os.path.join(swp.di, 'metric.pvd'))
local_norm = Constant(0.0)
for i, H in enumerate(average_hessians):
    local_norm.assign(pow(op.target, 2/d)*glob_norm*pow(timestep_integrals[i], -2/(2*p+d)))
    H.interpolate(local_norm*pow(det(H), -1/(2*p + d))*H)
    H.rename("Metric")
    metric_file.write(H)


# --- TODO: Adapt meshes


# --- TODO: Run Hessian based

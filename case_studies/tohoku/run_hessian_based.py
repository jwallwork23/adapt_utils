from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem
from adapt_utils.swe.utils import ShallowWaterHessianRecoverer
from adapt_utils.adapt.metric import metric_complexity

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
    target=float(args.target or 1.0e+03),  # Desired average instantaneous spatial complexity
)
op.end_time = float(args.end_time or op.end_time)

# Create problem object
swp = TsunamiProblem(op)


# --- Callbacks


recoverer = ShallowWaterHessianRecoverer(swp.V, op=op)  # L2 projector will be repeatedly used
hessian = lambda sol: recoverer.get_hessian_metric(sol, fields=swp.fields, noscale=True)
timestep = lambda sol: 1.0/op.dt


def extra_setup():

    # TODO: LaggedTimeIntegralCallback to reduce cost of Hessian computation
    # TODO: Option to take metric with maximum complexity, rather than time average

    # Number of timesteps per export (trivial for constant dt)
    swp.callbacks["timestep"] = callback.TimeIntegralCallback(
        timestep, swp.solver_obj, swp.solver_obj.timestepper, name="timestep", append_to_log=False)
    swp.solver_obj.add_callback(swp.callbacks["timestep"], 'timestep')

    # Time integrated Hessian over each window
    swp.callbacks["average_hessian"] = callback.TimeIntegralCallback(
        hessian, swp.solver_obj, swp.solver_obj.timestepper, name="average_hessian", append_to_log=False)
    swp.solver_obj.add_callback(swp.callbacks["average_hessian"], 'timestep')


swp.setup_solver_forward(extra_setup=extra_setup)


# --- Exports


average_hessian = Function(swp.P1_ten, name="Average Hessian")
average_hessians = []
timestep_integrals = []
hessian_file = File(os.path.join(swp.di, 'hessian.pvd'))


def export_func():
    if swp.solver_obj.iteration == 0:
        return

    # Extract time averaged Hessian and reset to zero for next window
    average_hessian.interpolate(swp.callbacks["average_hessian"].get_value())
    hessian_file.write(average_hessian)
    average_hessians.append(average_hessian.copy(deepcopy=True))
    swp.callbacks["average_hessian"].integrant = 0

    # Extract timesteps per export and reset to zero for next window
    timestep_integrals.append(swp.callbacks["timestep"].get_value())
    swp.callbacks["timestep"].integrant = 0.0


# --- Run fixed mesh


swp.solve(export_func=export_func)
print_output("Quantity of interest: {:.4e}".format(swp.callbacks["qoi"].get_value()))


# --- Time normalise metrics


time_normalise(average_hessians, timestep_integrals, op=op)
metric_file = File(os.path.join(swp.di, 'metric.pvd'))
complexities = []
for i, H in enumerate(average_hessians):
    metric_file.write(H)
    complexities.append(metric_complexity(H))


# --- Adapt meshes


meshes = []
for i, M in enumerate(average_hessians):
    mesh = adapt(swp.mesh, M)
    meshes.append(mesh)
# mesh_file = File(os.path.join(swp.di, 'mesh.pvd'))  # FIXME
for i, mesh in enumerate(meshes):
    msg = "{:2d}: complexity {:7.1f} vertices {:6d} elements {:6d}"
    print_output(msg.format(i, complexities[i], mesh.num_vertices(), mesh.num_cells()))
    mesh_file = File(os.path.join(swp.di, 'mesh_{:d}.pvd'.format(i)))  # FIXME
    mesh_file.write(mesh.coordinates)


# --- TODO: Run Hessian based

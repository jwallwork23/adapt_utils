from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem
from adapt_utils.adapt.metric import *

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

# Setup solver
swp = TsunamiProblem(op, levels=0)
swp.setup_solver_forward()

average_hessian = Function(swp.P1_ten, name="Average Hessian")
average_hessians = []
timestep_integrals = []
hessian_file = File(os.path.join(swp.di, 'hessian.pvd'))


# --- Callbacks


def hessian(sol):

    # TODO: Only setup L2 projection system once
    # TODO: Re-implement version of below which currently exists in swe/solver

    uv, elev = sol.split()
    if op.adapt_field == 'elevation':
        return steady_metric(elev, mesh=swp.mesh, noscale=True, op=op)
    elif op.adapt_field == 'speed':
        return steady_metric(sqrt(inner(uv, uv)), mesh=swp.mesh, noscale=True, op=op)
    elif op.adapt_field == 'elevation__int__speed':
        M_elev = steady_metric(elev, mesh=swp.mesh, noscale=True, op=op)
        M_spd = steady_metric(sqrt(inner(uv, uv)), mesh=swp.mesh, noscale=True, op=op)
        return metric_intersection(M_elev, M_spd)
    else:
        raise NotImplementedError  # TODO


timestep = lambda sol: 1.0/op.dt


def extra_setup():

    # TODO: LaggedTimeIntegralCallback to reduce cost of Hessian computation

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

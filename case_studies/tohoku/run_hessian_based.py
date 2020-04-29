from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem
from adapt_utils.swe.utils import ShallowWaterHessianRecoverer
from adapt_utils.adapt.metric import metric_complexity, time_normalise

import argparse


parser = argparse.ArgumentParser(prog="run_hessian_based")

# Timstepping
parser.add_argument("-end_time", help="End time of simulation (default 25 minutes)")

# Initial mesh
parser.add_argument("-level", help="(Integer) mesh resolution (default 0)")

# Adaptation
parser.add_argument("-num_meshes", help="Number of meshes to consider (default 5)")
parser.add_argument("-num_adapt", help="Number of iterations in adaptation loop (default 2)")
parser.add_argument("-norm_order", help="p for Lp normalisaton (default 1)")
parser.add_argument("-adapt_field", help="Field to construct metric w.r.t")
parser.add_argument("-target", help="Target space-time complexity (default 1.0e+04)")
parser.add_argument("-h_min", help="Minimum tolerated element size (default 100m)")
parser.add_argument("-h_max", help="Maximum tolerated element size (default 100km)")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()


# --- Setup


# Order for spatial Lp normalisation
# p = None
# if args.norm_order is not None:  # FIXME: L-inf time normalisation
#     p = float(args.norm_order)
#     assert p >= 1.0
p = 1

kwargs = {

    # Timestepping
    'end_time': float(args.end_time or 1500.0),

    # Initial mesh
    'level': int(args.level or 0),

    # Adaptation
    'approach': 'hessian',
    'num_meshes': int(args.num_meshes or 5),
    'num_adapt': int(args.num_adapt or 1),
    'norm_order': p,
    'adapt_field': args.adapt_field or 'elevation',
    'target': float(args.target or 1.0e+04),  # FIXME
    'h_min': float(args.h_min or 1.0e+02),
    'h_max': float(args.h_max or 1.0e+05),
    'plot_pvd': True,

    # Misc
    'debug': bool(args.debug or False),
}
paramstr = "\n"
logstr = 50*'*' + '\n' + 19*' ' + "PARAMETERS\n" + 50*'*' + '\n'
for key in kwargs:
    logstr += "    {:12s}: {:}\n".format(key, kwargs[key])
logstr +=  50*'*'
print_output(logstr)

# Create parameter class and problem object
op = TohokuOptions(**kwargs)
swp = AdaptiveTsunamiProblem(op)

average_hessians = [Function(P1_ten, name="Average Hessian") for P1_ten in swp.P1_ten]
timestep_integrals = np.zeros(swp.num_meshes)
hessian_file = File(os.path.join(swp.di, 'hessian.pvd'))


# --- Mesh adaptation loop


for n in range(op.num_adapt):
    if hasattr(swp, 'hessian_func'):
        delattr(swp, 'hessian_func')
    export_func = None

    for i in range(swp.num_meshes):

        # Transfer the solution from the previous mesh / apply initial condition
        swp.transfer_forward_solution(i)

        if n < op.num_adapt - 1:

            # Create double L2 projection operator which will be repeatedly used
            recoverer = ShallowWaterHessianRecoverer(swp.V[i], op=op)
            hessian = lambda sol: recoverer.get_hessian_metric(sol, fields=swp.fields[i], normalise=False)
            swp.hessian_func = hessian

            def export_func():

                # We only care about the final export in each mesh iteration
                if swp.fwd_solvers[i].iteration != (i+1)*swp.dt_per_mesh:
                    return

                # Extract time averaged Hessian
                average_hessians[i].interpolate(swp.callbacks[i]["average_hessian"].get_value())
                hessian_file.write(average_hessians[i])

                # Extract timesteps per mesh iteration
                timestep_integrals[i] = swp.callbacks[i]["timestep"].get_value()

        # Solve step for current mesh iteration
        swp.setup_solver_forward(i)
        swp.solve_forward_step(i, export_func=export_func)

        # TODO: Delete objects to free memory

    qoi = swp.quantity_of_interest()
    print_output("Quantity of interest: {:.4e}".format(qoi))
    swp.qois.append(qoi)
    if n == op.num_adapt - 1:
        break


    # --- Time normalise metrics


    time_normalise(average_hessians, timestep_integrals, op=op)
    metric_file = File(os.path.join(swp.di, 'metric.pvd'))
    complexities = []
    for i, H in enumerate(average_hessians):
        metric_file.write(H)
        complexities.append(metric_complexity(H))


    # --- Adapt meshes


    for i, M in enumerate(average_hessians):
        swp.meshes[i] = adapt(swp.meshes[i], M)
    swp.num_cells.append([mesh.num_cells() for mesh in swp.meshes])
    swp.num_vertices.append([mesh.num_vertices() for mesh in swp.meshes])

    print_output("\nAdaptation step {:2d}".format(n+1))
    for i, complexity in enumerate(complexities):
        msg = "  {:2d}: complexity {:8.1f} vertices {:7d} elements {:7d}"
        print_output(msg.format(i, complexity, swp.num_vertices[1][i], swp.num_cells[1][i]))
    print_output("")


    # ---  Setup for next run / logging


    swp.set_meshes(swp.meshes)
    swp.create_function_spaces()
    swp.dofs.append([np.array(V.dof_count).sum() for V in swp.V])
    swp.create_solutions()
    swp.set_fields()
    swp.set_stabilisation()
    swp.set_boundary_conditions()


# --- Print summary

logstr += 50*'*' + '\n' + 20*' ' + "SUMMARY\n" + 50*'*'
for i, qoi in enumerate(swp.qois):
    logstr += "Mesh iteration {:2d}: qoi {:.4e}".format(i, qoi)
logstr += 50*'*' 
print_output(logstr)

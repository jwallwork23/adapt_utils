from thetis import *

import argparse

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem


parser = argparse.ArgumentParser(prog="plot_dwp_indicators")

# Space-time domain
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-num_meshes", help="Number of meshes to consider")

# Solver
parser.add_argument("-family", help="Element family for mixed FE space")

# QoI
parser.add_argument("-start_time", help="Start time of period of interest")
parser.add_argument("-locations", help="""
Locations of interest, separated by commas. Choose from {'Fukushima Daiichi', 'Onagawa',
'Fukushima Daini', 'Tokai', 'Hamaoka', 'Tohoku', 'Tokyo'}.
""")
parser.add_argument("-radii", help="Radii of interest, separated by commas")

# Misc
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Collect locations and radii
if args.locations is None:
    locations = ['Fukushima Daiichi', ]
else:
    locations = args.locations.split(',')
if args.radii is None:
    radii = [100.0e+03 for l in locations]
else:
    radii = [float(r) for r in args.radii.split(',')]
if len(locations) != len(radii):
    msg = "Number of locations ({:d}) and radii ({:d}) do not match."
    raise ValueError(msg.format(len(locations), len(radii)))

# Set parameters for fixed mesh run
op = TohokuOptions(
    start_time=float(args.start_time or 0.0),
    end_time=float(args.end_time or 1440.0),
    # family=args.family or 'taylor-hood',
    family=args.family or 'dg-cg',
    level=int(args.level or 2),
    approach='fixed_mesh',
    plot_pvd=True,
    debug=bool(args.debug or False),
    num_meshes=int(args.num_meshes or 6),
    radii=radii,
    locations=locations,
)
assert op.start_time >= 0.0
assert op.start_time <= op.end_time

# Setup problem object
swp = AdaptiveTsunamiProblem(op)

# Solve forward to get checkpoints
swp.solve_forward()

# Loop over mesh windows *in reverse*
indicators = [Function(P1, name="DWP indicator") for P1 in swp.P1]
for i in range(swp.num_meshes-1, -1, -1):

    # --- Solve adjoint on current window

    adj_solutions_step = []

    def export_func():
        adj_solutions_step.append(swp.adj_solutions[i].copy(deepcopy=True))

    swp.transfer_adjoint_solution(i)
    swp.setup_solver_adjoint(i)
    swp.solve_adjoint_step(i, export_func=export_func)

    # --- Solve forward on current window

    fwd_solutions_step = []

    def export_func():
        fwd_solutions_step.append(swp.fwd_solvers[i].fields.solution_2d.copy(deepcopy=True))

    swp.transfer_forward_solution(i)
    swp.setup_solver_forward(i)
    swp.solve_forward_step(i, export_func=export_func)

    # --- Assemble indicators

    n = len(fwd_solutions_step)
    n_adj = len(adj_solutions_step)
    if n != n_adj:
        raise ValueError("Mismatching number of indicators ({:d} vs {:d})".format(n, n_adj))
    I = 0
    op.print_debug("DWP indicators on mesh {:2d}".format(i))
    for j, solutions in enumerate(zip(fwd_solutions_step, reversed(adj_solutions_step))):
        scaling = 0.5 if j in (0, n-1) else 1.0  # Trapezium rule  # TODO: Other integrators
        fwd_dot_adj = abs(inner(*solutions))
        op.print_debug("    ||<q, q*>||_L2 = {:.4e}".format(assemble(fwd_dot_adj*fwd_dot_adj*dx)))
        I += op.dt*swp.dt_per_mesh*scaling*fwd_dot_adj
    indicators[i].interpolate(I)

for indicator in indicators:
    swp.indicator_file._topology = None
    swp.indicator_file.write(indicator)

import argparse

from adapt_utils.swe.tsunami.solver import AdaptiveTsunamiProblem
from adapt_utils.case_studies.tohoku.options import TohokuOptions


parser = argparse.ArgumentParser(prog="run_continuous_adjoint")
parser.add_argument("-level", help="(Integer) mesh resolution")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-start_time")  # TODO: doc
parser.add_argument("-num_meshes", help="Number of meshes to consider")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-radius")  # TODO: doc
parser.add_argument("-family")  # TODO: doc
args = parser.parse_args()

# Set parameters for fixed mesh run
op = TohokuOptions(
    family=args.family or 'taylor-hood',
    level=int(args.level or 0),
    approach='fixed_mesh',
    plot_pvd=True,
    debug=bool(args.debug or False),
    num_meshes=int(args.num_meshes or 1),
    radii=[float(args.radius or 50.0e+03), ],  # TODO: Other locations and radii
)
op.adjoint_params = {
    "ksp_type": "gmres",
    # "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
}
op.start_time = float(args.start_time or op.start_time)
op.end_time = float(args.end_time or op.end_time)

# Solve
swp = AdaptiveTsunamiProblem(op)
swp.solve_adjoint()

from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse

parser = argparse.ArgumentParser(prog="run_fixed_mesh")
parser.add_argument("-num_initial_adapt", type=int, help="Number of initial adaptation steps")
parser.add_argument("-n", type=int, help="Initial uniform mesh resolution")
parser.add_argument("-end_time", help="End time of simulation")
parser.add_argument("-adapt_field", help="Field to adapt w.r.t. Choose multiple seperated by '__'")
parser.add_argument("-approach", help="Mesh adaptation approach. Choose multiple seperated by '__'")
parser.add_argument("-target", help="Target mesh complexity/error for metric based methods")
parser.add_argument("-alpha", type=float,
                    help="Tuning parameter for monitor functions related to magnitude")
parser.add_argument("-beta", type=float,
                    help="Tuning parameter for monitor functions related to scale")
parser.add_argument("-r_adapt_rtol", type=float, help="Relative tolerance for r-adaptation.")
parser.add_argument("-debug", help="Print all debugging statements")
args = parser.parse_args()

# Read parameters
n = args.n or 40
num_adapt = args.num_initial_adapt or 0
offset = 0  # TODO: Use offset
alpha = args.alpha or 0.001
beta = args.beta or 0.005
adapt_fields = (args.adapt_field or 'bathymetry').split('__')
approaches = (args.approach or 'fixed_mesh').split('__')
if len(adapt_fields) < len(approaches):
    adapt_fields = [adapt_fields[0] for approach in approaches]
elif len(adapt_fields) > len(approaches):
    approaches = [approach[0] for field in adapt_fields]

# Adapt mesh in lonlat space
ext = None
if num_adapt > 0:
    ext = "{:s}_{:d}".format('_'.join(adapt_fields), num_adapt)  # FIXME
    op_init = TohokuOptions(utm=False, n=n, offset=offset, nonlinear_method='quasi_newton',
                            h_max=1.0e+10, target=float(args.target or 1.0e+4),
                            debug=bool(args.debug or False), r_adapt_rtol=args.r_adapt_rtol or 0.01)
    for approach, adapt_field in zip(approaches, adapt_fields):
        op_init.adapt_field = adapt_field
        if not hasattr(op_init, 'lonlat_mesh'):
            op_init.get_lonlat_mesh()
        swp_init = TsunamiProblem(op_init, levels=0)
        swp_init.initialise_mesh(approach=approach or 'hessian', alpha=alpha, beta=beta, num_adapt=num_adapt)
        op_init.default_mesh = swp_init.mesh
op_init.latlon_mesh = swp_init.mesh
op_init.get_utm_mesh()

# Set parameters for fixed mesh run
op = TohokuOptions(mesh=op_init.default_mesh, utm=True, plot_pvd=True, offset=offset)
op.end_time = float(args.end_time or op.end_time)
op.get_lonlat_mesh()
op.set_bathymetry()

# Set wetting and drying parameter
# op.wetting_and_drying_alpha.assign(0.5)
h = CellSize(op.default_mesh)
b = op.bathymetry
P0 = FunctionSpace(op.default_mesh, "DG", 0)  # NOTE: alpha is enormous in this approach (O(km))
op.wetting_and_drying_alpha = interpolate(h*sqrt(dot(grad(b), grad(b))), P0)

# Solve
swp = TsunamiProblem(op, levels=0, extension=ext)
swp.solve()
op.plot_timeseries("P02", extension=ext)
op.plot_timeseries("P06", extension=ext)

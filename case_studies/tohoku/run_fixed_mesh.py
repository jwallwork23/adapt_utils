from thetis import *

from adapt_utils.case_studies.tohoku.options import TohokuOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-num_initial_adapt")
parser.add_argument("-n")
parser.add_argument("-end_time")
parser.add_argument("-adapt_field")
parser.add_argument("-approach")
parser.add_argument("-target")
parser.add_argument("-alpha")
parser.add_argument("-beta")
parser.add_argument("-debug")
args = parser.parse_args()

n = int(args.n or 40)
num_adapt = int(args.num_initial_adapt or 0)
offset = 0  # TODO: Use offset
alpha = float(args.alpha or 0.001)
beta = float(args.beta or 0.005)


# Adapt mesh in lonlat space
ext = None
adapt_field = args.adapt_field or 'bathymetry'
adapt_fields = adapt_field.split('__')
approach = args.approach or 'fixed_mesh'
approaches = approach.split('__')
R = zip(approaches, [adapt_fields[0] for f in approaches]) if len(adapt_fields) < len(approaches) else zip(approaches, adapt_fields)
if num_adapt > 0:
    ext = "{:s}_{:d}".format(adapt_field, num_adapt)  # FIXME
    op_init = TohokuOptions(utm=False, n=n, offset=offset, nonlinear_method='quasi_newton',
                            h_max=1.0e+10, target=float(args.target or 1.0e+4),
                            debug=bool(args.debug or False), r_adapt_rtol=1.0e-2)
    for approach, field in R:
        op_init.adapt_field=field
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

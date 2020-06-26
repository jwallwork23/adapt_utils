from thetis import *
# from firedrake_adjoint import *

import os
import numpy as np
import argparse
# import matplotlib.pyplot as plt

from adapt_utils.test_cases.solid_body_rotation.options import LeVequeOptions
from adapt_utils.solver import AdaptiveProblem
# from adapt_utils.solver_discrete import AdaptiveDiscreteAdjointProblem


def write(text, out):
    out.write(text+'\n')
    print_output(text)


# Initialise
parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh refinement level")
parser.add_argument("-geometry", help="Choose from 'circle' or 'square'")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
parser.add_argument("-limiters", help="Toggle limiters for tracer equation")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose FEM from 'cg' and 'dg'")
# parser.add_argument("-compute_gradient", help="Toggle gradient computation")
args = parser.parse_args()

i = int(args.level or 0)
n = 2**i
geometry = args.geometry or 'circle'

# Create parameter class
kwargs = {
    'approach': 'fixed_mesh',

    # Geometry
    'geometry': geometry,

    # Spatial discretisation
    'refinement_level': i,
    'tracer_family': args.family or 'dg',
    'stabilisation': args.stabilisation or 'sipg',
    'use_automatic_sipg_parameter': False,
    'use_limiter_for_tracers': bool(args.limiters or True),
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Temporal discretisation
    'dt': np.pi/(100*n),
    # 'dt': np.pi/(300*n),
    'dt_per_export': 10*n,
}
op = LeVequeOptions(**kwargs)
print_output("Element count: {:d}".format(op.default_mesh.num_cells()))


# class TracerProblem(AdaptiveDiscreteAdjointProblem):
class TracerProblem(AdaptiveProblem):

    def quantity_of_interest(self):
        kernel = self.op.set_qoi_kernel(self.P0[-1])
        sol = self.fwd_solutions_tracer[-1]
        return assemble(kernel*sol*dx)


# Run model
tp = TracerProblem(op)
tp.solve()

# Print outputs
f = open(os.path.join(tp.di, "{:s}_{:d}.log".format(geometry, n)), 'w+')
head = "\n  Shape            Analytic QoI    Quadrature QoI  Calculated QoI  Error"
rule = 74*'='
write(head, f)
write(rule, f)
for shape, name in zip(range(3), ('Gaussian', 'Cone', 'Slotted cylinder')):
    op.shape = shape
    exact = op.exact_qoi()
    qoi = tp.quantity_of_interest()
    qois = (exact, op.quadrature_qoi(tp.P0[0]), qoi, 100.0*abs(1.0 - qoi/exact))
    line = "{:1d} {:16s} {:14.8e}  {:14.8e}  {:14.8e}  {:6.4f}%".format(shape, name, *qois)
    write(line, f)
write(rule, f)
approach = "'" + tp.approach.replace('_', ' ').capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7d}"
write(tail.format('Approach', approach, 'Element count', tp.num_cells[-1][-1]), f)
geometry = "'" + geometry.capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7.4f}"
write(tail.format('Geometry', geometry, 'Timestep', op.dt), f)
f.close()  # TODO: Print model parameters, too

# # Compute gradient w.r.t. velocity
# if bool(args.compute_gradient or False):
#     print_output("Computing gradient...")
#     u = Control(tp.fwd_solutions[0])       # fluid velocity - elevation tuple
#     g = tp.compute_gradient(u).split()[0]  # gradient w.r.t. velocity
#     tricontourf(g)
#     plt.title("Gradient of QoI w.r.t. fluid speed")
#     plt.show()

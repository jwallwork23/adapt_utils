from thetis import *

import os
import numpy as np
import argparse

from adapt_utils.test_cases.solid_body_rotation.options import LeVequeOptions
from adapt_utils.adapt.solver import AdaptiveProblem


def write(text, out):
    out.write(text+'\n')
    print_output(text)


# Initialise
parser = argparse.ArgumentParser()
parser.add_argument("-level", help="Mesh refinement level")
parser.add_argument("-geometry", help="Choose from 'circle' or 'square'")
parser.add_argument("-conservative", help="Toggle conservative tracer equation")
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
    'tracer_family': 'dg',
    # 'stabilisation': 'no',
    'stabilisation': 'sipg',
    'use_automatic_sipg_parameter': False,
    'use_limiter_for_tracers': True,
    'use_tracer_conservative_form': bool(args.conservative or False),

    # Temporal discretisation
    'dt': np.pi/(100*n),
    # 'dt': np.pi/(300*n),
    'dt_per_export': 10*n,
}
op = LeVequeOptions(**kwargs)
print_output("Element count: {:d}".format(op.default_mesh.num_cells()))

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

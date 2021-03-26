from thetis import *

import argparse
import os

from adapt_utils.unsteady.test_cases.solid_body_rotation.options import LeVequeOptions
from adapt_utils.unsteady.solver import AdaptiveProblem


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
parser.add_argument("-family", help="Choose finite element from 'cg' and 'dg'")
args = parser.parse_args()

i = int(args.level or 0)
n = 2**i
geometry = args.geometry or 'circle'

# Create parameter class
kwargs = {
    'approach': 'lagrangian',

    # Geometry
    'geometry': geometry,

    # Spatial discretisation
    'level': i,
    'tracer_family': args.family or 'dg',
    'stabilisation_tracer': args.stabilisation,
    'use_automatic_sipg_parameter': False,  # We have an inviscid problem
    'use_limiter_for_tracers': bool(args.limiters or True),
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
        kernel = self.op.set_qoi_kernel_tracer(self, -1)
        sol = self.fwd_solutions_tracer[-1]
        return assemble(kernel*sol*dx(degree=12))


# Run model
tp = TracerProblem(op)
init_coords = tp.meshes[0].coordinates.dat.data.copy()
tp.solve_forward()
final_coords = tp.meshes[0].coordinates.dat.data
assert np.allclose(init_coords, final_coords, atol=1.0e-02), "Non-matching initial and final mesh coords"

# Print outputs
f = open(os.path.join(tp.di, "{:s}_{:d}.log".format(geometry, n)), 'w+')
head = "\n  Shape            Analytic QoI    Quadrature QoI  Calculated QoI  Error    Disc. error"
rule = 90*'='
write(head, f)
write(rule, f)
for shape, name in zip(range(3), ('Gaussian', 'Cone', 'Slotted cylinder')):
    op.set_region_of_interest(shape)
    tp.op.set_qoi_kernel_tracer(tp, -1)
    qoi = tp.quantity_of_interest()
    exact = op.exact_qoi()
    quadrature = op.quadrature_qoi(tp, -1)
    qois = (exact, quadrature, qoi, 100*abs(1.0 - qoi/exact), 100*abs(1.0 - qoi/quadrature))
    line = "{:1d} {:16s} {:14.8e}  {:14.8e}  {:14.8e}  {:6.4f}%  {:6.4f}%".format(shape, name, *qois)
    write(line, f)
write(rule, f)
approach = "'" + tp.approach.replace('_', ' ').capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7d}"
write(tail.format('Approach', approach, 'Element count', tp.num_cells[-1][-1]), f)
geometry = "'" + geometry.capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7.4f}"
write(tail.format('Geometry', geometry, 'Timestep', op.dt), f)
f.close()  # TODO: Print model parameters, too

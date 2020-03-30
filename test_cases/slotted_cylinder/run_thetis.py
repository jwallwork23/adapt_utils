from thetis import print_output

import os
import numpy as np
import argparse

from adapt_utils.test_cases.slotted_cylinder.options import LeVequeOptions
from adapt_utils.tracer.solver2d_thetis import UnsteadyTracerProblem2d_Thetis


def write(text, out):
    out.write(text+'\n')
    print_output(text)

# Initialise
parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-solve_adjoint", help="Solve adjoint problem and save to hdf5")
parser.add_argument("-init_res", help="Initial mesh refinement level")
parser.add_argument("-num_adapt", help="Number of adapt/solve iterations in adaptation loop")
parser.add_argument("-desired_error", help="Desired error level in metric normalisation")
parser.add_argument("-mesh_type", help="Choose from 'circle' or 'square'.")
parser.add_argument("-shape", help="""
Shape defining QoI, chosen from:
  0: Gaussian bell
  1: Cone
  2: Slotted cylinder
""")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
adj = bool(args.solve_adjoint or False)
i = int(args.init_res or 0)
n = 2**i
desired_error = float(args.desired_error or 0.1)
mesh_type = args.mesh_type or 'circle'

# Create parameter class
kwargs = {
    'mesh_type': mesh_type,

    # QoI  (default slotted cylinder)
    'shape': int(args.shape or 2),

    # Spatial discretisation
    'refinement_level': i,
    'family': 'dg',
    'stabilisation': 'no',

    # Temporal discretisation
    'dt': np.pi/(100*n),
    # 'dt': np.pi/(300*n),
    'dt_per_export': 10*n,
    'dt_per_remesh': 10*n,

    # Adaptation parameters
    'approach': approach,
    'normalisation': 'error',
    'num_adapt': int(args.num_adapt or 3),
    'norm_order': 1,
    'desired_error': desired_error,
    'target': 1.0/desired_error,
}
op = LeVequeOptions(**kwargs)

# Run model
tp = UnsteadyTracerProblem2d_Thetis(op=op)
if approach == 'hessian':
    tp.adapt_mesh()
    tp.set_initial_condition()
    tp.set_fields()
elif adj:
    op.approach = 'fixed_mesh'
    tp_ = UnsteadyTracerProblem2d_Thetis(op=op)
    tp_.solve()
    tp_.solve_adjoint()
    op.approach = args.approach
# TODO: initial solves and adapts to get good initial mesh
tp.solve()

# Print outputs
f = open(os.path.join(tp.di, "{:s}_{:d}.log".format(mesh_type, n)), 'w+')
head = "\n  Shape            Analytic QoI    Quadrature QoI  Calculated QoI  Error"
rule = 74*'='
write(head, f)
write(rule, f)
for shape, name in zip(range(3), ('Gaussian', 'Cone', 'Slotted cylinder')):
    op.set_region_of_interest(shape=shape)
    tp.get_qoi_kernel()
    exact = op.exact_qoi()
    qoi = tp.quantity_of_interest()
    qois = (exact, op.quadrature_qoi(tp.P0), qoi, 100.0*abs(1.0 - qoi/exact))
    line = "{:1d} {:16s} {:14.8e}  {:14.8e}  {:14.8e}  {:6.4f}%".format(shape, name, *qois)
    write(line, f)
write(rule, f)
approach = "'" + tp.approach.replace('_', ' ').capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7d}"
write(tail.format('Approach', approach, 'Element count', tp.num_cells[-1]), f)
mesh_type = "'" + mesh_type.capitalize() + "'"
tail = 19*' ' + "{:14s}  {:14s}  {:14s}  {:7.4f}"
write(tail.format('Mesh type', mesh_type, 'Timestep', op.dt), f)
f.close()

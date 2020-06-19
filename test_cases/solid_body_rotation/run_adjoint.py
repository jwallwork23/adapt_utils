"""
A simple demo to show that discrete adjoint works.
"""
from thetis import *
from firedrake_adjoint import *

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from adapt_utils.test_cases.solid_body_rotation.options import LeVequeOptions
from adapt_utils.adapt.solver import AdaptiveProblem


def write(text, out):
    out.write(text+'\n')
    print_output(text)


# Initialise
parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Choose adaptive approach")
parser.add_argument("-init_res", help="Initial mesh refinement level")
parser.add_argument("-mesh_type", help="Choose from 'circle' or 'square'")
parser.add_argument("-shape", help="""
Shape defining QoI, chosen from:
  0: Gaussian bell
  1: Cone
  2: Slotted cylinder
""")
args = parser.parse_args()

approach = args.approach or 'fixed_mesh'
i = int(args.init_res or 0)
n = 2**i
mesh_type = args.mesh_type or 'circle'

# Create parameter class
kwargs = {
    'approach': 'fixed_mesh',
    'mesh_type': mesh_type,

    # QoI  (default slotted cylinder)
    'shape': int(args.shape or 2),

    # Spatial discretisation
    'refinement_level': i,
    'tracer_family': 'dg',
    'stabilisation': 'no',

    # Temporal discretisation
    'dt': np.pi/(100*n),
    'dt_per_export': 10*n,
    'dt_per_remesh': 10*n,
}
op = LeVequeOptions(**kwargs)
print_output("Element count: {:d}".format(op.default_mesh.num_cells()))

# Run model
tp = AdaptiveProblem(op)
tp.solve()

J = assemble(tp.fwd_solutions_tracer[0]*tp.fwd_solutions_tracer[0]*dx)
g = compute_gradient(J, Control(tp.fwd_solutions[0])).split()[0]
tricontourf(g)
plt.title("Gradient of QoI w.r.t. fluid speed")
plt.show()

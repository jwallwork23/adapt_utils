from thetis import *
from firedrake_adjoint import *

import argparse
import os

from adapt_utils.unsteady.test_cases.solid_body_rotation.options import LeVequeOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


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
parser.add_argument("-debug", help="Toggle debugging")
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

    # Misc
    'debug': bool(args.debug or False),
}
op = LeVequeOptions(**kwargs)
di = op.di
print_output("Element count: {:d}".format(op.default_mesh.num_cells()))


class AnnotatedTracerProblem(AdaptiveDiscreteAdjointProblem):

    def quantity_of_interest(self):
        kernel = self.op.set_qoi_kernel_tracer(self, -1)
        sol = self.fwd_solutions_tracer[-1]
        return assemble(kernel*sol*dx)


# Solve forward
op.di = create_directory(os.path.join(di, 'discrete'))
tp_discrete = AnnotatedTracerProblem(op)
print_output("Running forward model...")
tp_discrete.solve_forward()

# # Solve discrete adjoint  # FIXME
# print_output("Running discrete adjoint model...")
# tp_discrete.solve_adjoint()
# tp_discrete.save_adjoint_trajectory()
stop_annotating()


class UnannotatedTracerProblem(AdaptiveProblem):

    def quantity_of_interest(self):
        kernel = self.op.set_qoi_kernel_tracer(self, -1)
        sol = self.fwd_solutions_tracer[-1]
        return assemble(kernel*sol*dx)


# Solve continuous adjoint
op.di = create_directory(os.path.join(di, 'continuous'))
tp_continuous = UnannotatedTracerProblem(op)
tp_continuous.set_initial_condition()  # Adjoint propagation is driven by reverse flow
print_output("Running continuous adjoint model...")
tp_continuous.solve_adjoint()

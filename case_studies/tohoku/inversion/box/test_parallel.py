from thetis import *
from firedrake_adjoint import *

import argparse
import numpy as np
import os

from adapt_utils.case_studies.tohoku.options.box_options import TohokuBoxBasisOptions
from adapt_utils.unsteady.solver_adjoint import AdaptiveDiscreteAdjointProblem


nproc = COMM_WORLD.size


# --- Parse arguments

parser = argparse.ArgumentParser()

# Model
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-family", help="Finite element pair")
parser.add_argument("-stabilisation", help="Stabilisation approach")
parser.add_argument("-nonlinear", help="Toggle nonlinear model")

# Inversion
parser.add_argument("-noisy_data", help="Toggle whether to sample noisy data")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries")

# I/O and debugging
parser.add_argument("-debug", help="Toggle debugging")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
timeseries_type = "timeseries"
if bool(args.continuous_timeseries or False):
    timeseries_type = "_".join([timeseries_type, "smooth"])

# Collect initialisation parameters
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': args.stabilisation,

    # Inversion
    'synthetic': False,
    'qoi_scaling': 1.0,
    'noisy_data': bool(args.noisy_data or False),

    # I/O and debugging
    'plot_pvd': True,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
nonlinear = bool(args.nonlinear or False)

# Setup output directories
dirname = os.path.dirname(__file__)
di = create_directory(os.path.join(dirname, 'outputs', 'realistic', 'discrete'))

# Load control parameters
fname = os.path.join(di, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
kwargs['control_parameters'] = np.load(fname.format('ctrl', level))[-1]
op = TohokuBoxBasisOptions(**kwargs)
gauges = list(op.gauges.keys())
op.di = create_directory(os.path.join(dirname, 'outputs', 'test_parallel', str(nproc)))


# --- Solve

# Solve the forward problem with initial guess
op.save_timeseries = True
print_output("Run forward...")
swp = AdaptiveDiscreteAdjointProblem(op, nonlinear=nonlinear, print_progress=op.debug)
swp.solve_forward()
J = swp.quantity_of_interest()
print_output("Quantity of interest = {:.8e}".format(J))

# Solve adjoint problem and plot solution fields
print_output("Run adjoint...")
swp.solve_adjoint()
# swp.compute_gradient(Control(op.control_parameters[0]))  # TODO: Use solve_adjoint
swp.get_solve_blocks()
swp.save_adjoint_trajectory()

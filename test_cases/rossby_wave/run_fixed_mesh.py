from thetis import *

import argparse

from adapt_utils.test_cases.rossby_wave.options import BoydOptions
from adapt_utils.test_cases.rossby_wave.monitors import *
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


parser = argparse.ArgumentParser()
parser.add_argument("-n_coarse", help="Resolution of coarse mesh.")
parser.add_argument("-n_fine", help="Resolution of fine mesh.")
parser.add_argument("-end_time", help="Simulation end time.")
parser.add_argument("-refine_equator", help="""
Apply Monge-Ampere based r-adaptation to refine equatorial region.""")
parser.add_argument("-calculate_metrics", help="Compute metrics using the fine mesh.")
parser.add_argument("-read_only", help="Read error file instead of computing anew.")
parser.add_argument("-debug", help="Toggle debugging mode.")
args = parser.parse_args()


n_coarse = int(args.n_coarse or 1)  # NOTE: [Huang et al 2008] considers n = 4, 8, 20
n_fine = int(args.n_fine or 50)
refine_equator = bool(args.refine_equator or False)
initial_monitor = equator_monitor if refine_equator else None  # TODO: Other options
read_only = bool(args.read_only or False)

kwargs = {
  'n': n_coarse,
  'debug': bool(args.debug or False),
  'end_time': float(args.end_time or 120.0),
  'dt': 0.04/n_coarse,
  'plot_pvd': n_coarse < 5,
  'dt_per_export': 50*n_coarse,
  'order': 1,
  # 'order': 0,
  'num_adapt': 1,
  'r_adapt_rtol': 1.0e-3,
}

op = BoydOptions(**kwargs)
swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.setup_solver()

if initial_monitor is not None:
    swp.approach = 'monge_ampere'
    swp.monitor_function = initial_monitor
    swp.adapt_mesh()
    # op.approach = 'fixed_mesh'  # TODO: check if needed
    swp.__init__(op, mesh=swp.mesh, levels=swp.levels)

fname = '{:s}_{:d}'.format("uniform" if initial_monitor is None else "refined_equator", n_coarse)
if not read_only:
    swp.solve(uses_adjoint=False)
    swp.op.write_to_hdf5(fname)
swp.op.plot_errors()

if bool(args.calculate_metrics or False):
    print_output("\nCalculating error metrics...")
    op.get_peaks(swp.solution.split()[1], reference_mesh_resolution=n_fine)
    print_output("h+       : {:.8e}".format(op.h_upper))
    print_output("h-       : {:.8e}".format(op.h_lower))
    print_output("C+       : {:.8e}".format(op.c_upper))
    print_output("C-       : {:.8e}".format(op.c_lower))
    print_output("RMS error: {:.8e}".format(op.rms))

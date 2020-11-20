from thetis import create_directory, Constant, File

import argparse
import numpy as np
import os

from adapt_utils.steady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.swe.turbine.solver import AdaptiveTurbineProblem


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-level', help="""
    Number of uniform refinements to apply to the initial mesh (default 0)""")
parser.add_argument('-debug', help="Toggle debugging mode (default False).")
parser.add_argument('-debug_mode', help="""
    Choose debugging mode from 'basic' and 'full' (default 'basic').""")
args = parser.parse_args()


# Set parameters
kwargs = {
    'approach': 'fixed_mesh',
    'level': int(args.level or 1),
    'box': True,
    'plot_pvd': True,
    'debug': False if args.debug == "0" else True,
    'debug_mode': args.debug_mode or 'basic',
}
discrete_turbines = True
# discrete_turbines = False
op = TurbineArrayOptions(**kwargs)
op.update({
    'spun': False,
    'di': create_directory(os.path.join(op.di, 'unsteady')),

    # Extend to time-dependent case
    'timestepper': 'CrankNicolson',
    'dt': 5.0,
    'dt_per_export': 1,
    'end_time': 600.0,

    # Crank down viscosity and plot vorticity
    'base_viscosity': 0.00005,
    'characteristic_velocity': Constant(op.inflow_velocity),
    'grad_depth_viscosity': True,
    'max_reynolds_number': 10000.0,
    'recover_vorticity': True,

    # Only consider the first turbine
    'region_of_interest': [op.region_of_interest[0]],
    'array_ids': np.array([2]),
    'farm_ids': (2,),
})

# Solve forward problem
tp = AdaptiveTurbineProblem(op, discrete_turbines=discrete_turbines, ramp_dir=op.di)
File(os.path.join(op.di, 'viscosity.pvd')).write(tp.fields[0].horizontal_viscosity)
tp.solve_forward()

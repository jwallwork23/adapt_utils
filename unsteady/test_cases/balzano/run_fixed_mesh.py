"""
Test case created in [Balzano].

[Balzano] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in
shallow water flow models." Coastal Engineering 34.1-2 (1998): 83-107.
"""
from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type", help="""
Choose bathymetry type from {1, 2, 3}. Option 1 corresponds to a linear bathymetry, whereas 2 and 3
have kinks. See [Balzano] for details.
""")
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg-cg', 'dg-cg' and 'dg-dg'")
parser.add_argument("-debug", help="Toggle debugging")
args = parser.parse_args()

stabilisation = args.stabilisation or 'lax_friedrichs'

kwargs = {
    'approach': 'fixed_mesh',
    'n': 2,

    # Geometry
    'bathymetry_type': int(args.bathymetry_type or 1),

    # Spatial discretisation
    'family': args.family or 'dg-cg',
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': True,

    # Misc
    'plot_pvd': True,
    'debug': bool(args.debug or False),
}

op = BalzanoOptions(**kwargs)
swp = AdaptiveProblem(op)
swp.solve_forward()

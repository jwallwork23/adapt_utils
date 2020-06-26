from adapt_utils.unsteady.test_cases.balzano.options import BalzanoOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-bathymetry_type")  # TODO: doc
parser.add_argument("-stabilisation", help="Stabilisation method")
parser.add_argument("-family", help="Choose finite element from 'cg-cg', 'dg-cg' and 'dg-dg'")
args = parser.parse_args()

stabilisation = args.stabilisation or 'lax_friedrichs'

kwargs = {
    'approach': 'fixed_mesh',

    # Geometry
    'bathymetry_type': int(args.bathymetry_type or 1),

    # Spatial discretisation
    'family': args.family or 'dg-dg',
    'stabilisation': stabilisation,
    'use_automatic_sipg_parameter': True,
}

op = BalzanoOptions(**kwargs)
swp = AdaptiveProblem(op)
swp.solve_forward()

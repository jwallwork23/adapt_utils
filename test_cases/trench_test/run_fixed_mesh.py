from thetis import *

from adapt_utils.test_cases.trench_test.hydro_options import TrenchHydroOptions, export_final_state
from adapt_utils.swe.tsunami.solver import TsunamiProblem

op = TrenchHydroOptions(debug=False,
                    friction = 'nikuradse',
                    nx=1,
                    ny = 1)

tp = TsunamiProblem(op, levels=0)
tp.solve(uses_adjoint=False)

uv, elev = tp.solution.split()

export_final_state("hydrodynamics_trench", uv, elev)
from adapt_utils.test_cases.trench_test.hydro_options import TrenchHydroOptions, export_final_state
from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


op = TrenchHydroOptions(debug=False,
                        friction='nikuradse',
                        nx=1,
                        ny=1)

swp = UnsteadyShallowWaterProblem(op, levels=0)
swp.solve(uses_adjoint=False)

export_final_state("hydrodynamics_trench", *swp.solution.split())

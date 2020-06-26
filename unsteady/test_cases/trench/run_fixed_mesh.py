from adapt_utils.unsteady.test_cases.trench.hydro_options import TrenchHydroOptions, export_final_state
from adapt_utils.unsteady.solver import AdaptiveProblem


op = TrenchHydroOptions(debug=False, friction='nikuradse', nx=1, ny=1)
swp = AdaptiveProblem(op)
swp.solve_forward()
export_final_state("hydrodynamics_trench", *swp.fwd_solutions[0].split())

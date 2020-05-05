from adapt_utils.test_cases.trench_test.hydro_options import TrenchHydroOptions, export_final_state
from adapt_utils.adapt.solver import AdaptiveProblem


op = TrenchHydroOptions(debug=False, friction='nikuradse', nx=1, ny=1)
swp = AdaptiveProblem(op, levels=0)
swp.solve_forward()
export_final_state("hydrodynamics_trench", *swp.fwd_solutions[0].split())

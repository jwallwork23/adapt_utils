from firedrake import *

from time import perf_counter

from adapt_utils.steady.solver import AdaptiveSteadyProblem
from adapt_utils.steady.test_cases.point_discharge2d.options import PointDischarge2dOptions


# Solve forward problem
op = PointDischarge2dOptions(approach='dwr', level=2)
op.tracer_family = 'cg'                  # TODO: 'dg'
op.stabilisation_tracer = 'supg'         # TODO: 'lax_friedrichs'
op.anisotropic_stabilisation = True      # TODO: False
op.use_automatic_sipg_parameter = False  # TODO: True
tp = AdaptiveSteadyProblem(op)
tp.solve_forward()

# Compute error
c = op.analytical_solution(tp.Q[0])
c_h = tp.fwd_solution_tracer
e = c.copy(deepcopy=True)
e -= c_h

# Evaluate QoI and create effectivity index function
Je = assemble(op.set_qoi_kernel(op.default_mesh)*e*dx(degree=12))
I_eff = lambda eta: abs(eta/Je)

# TODO: Global enrichment using hp-refinement
# tic = perf_counter()
# GE_hp_indicator = tp.dwr_indicator('tracer', mode='GE_hp')
# GE_hp = tp.estimators['dwr'][0]
# GE_hp_time = perf_counter() - tic

# Global enrichment using h-refinement only
tic = perf_counter()
GE_h_indicator = tp.dwr_indicator('tracer', mode='GE_h')
GE_h = tp.estimators['dwr'][0]
GE_h_time = perf_counter() - tic

# TODO: Global enrichment using p-refinement only
# tic = perf_counter()
# GE_p_indicator = tp.dwr_indicator('tracer', mode='GE_p')
# GE_p = tp.estimators['dwr'][0]
# GE_p_time = perf_counter() - tic

# TODO: Local enrichment using p-refinement only
# tic = perf_counter()
# LE_p_indicator = tp.dwr_indicator('tracer', mode='LE_p')
# LE_p = tp.estimators['dwr'][0]
# LE_p_time = perf_counter() - tic

# TODO: Difference quotients
# tic = perf_counter()
# DQ_indicator = tp.dwr_indicator('tracer', mode='DQ')
# DQ = tp.estimators['dwr'][0]
# DQ_time = perf_counter() - tic

print("J(e)  = {:11.4e}".format(Je))
# print("GE_hp = {:11.4e}  I_eff = {:11.4e}  time = {:5.2f}".format(GE_hp, I_eff(GE_hp), GE_hp_time))
print("GE_h  = {:11.4e}  I_eff = {:11.4e}  time = {:5.2f}".format(GE_h, I_eff(GE_h), GE_h_time))
# print("GE_p = {:11.4e}  I_eff = {:11.4e}  time = {:5.2f}".format(GE_p, I_eff(GE_p), GE_p_time))
# print("LE_p = {:11.4e}  I_eff = {:11.4e}  time = {:5.2f}".format(LE_p, I_eff(LE_p), LE_p_time))
# print("DQ   = {:11.4e}  I_eff = {:11.4e}  time = {:5.2f}".format(DQ, I_eff(DQ), DQ_time))

from firedrake import *

from tracer.solver import TracerProblem

num_adapt = 4
nx = 3

for mode in ('hessian', 'explicit', 'dwp', 'dwr', 'dwr_adjoint', 'dwr_both', 'dwr_relaxed', 'dwr_superposed'):
    for i in range(num_adapt):
        if i == 0:
            tp = TracerProblem(stab='SUPG', n=nx)
        else:
            tp = TracerProblem(stab='SUPG', mesh=tp.mesh)
        #tp = TracerProblem(stab='SUPG', n=nx) if i == 0 else TracerProblem(stab='SUPG', mesh=tp.mesh)
        tp.solve()
        if mode not in ('hessian', 'explicit'):
            tp.solve_adjoint()
        tp.adapt_mesh(mode=mode)

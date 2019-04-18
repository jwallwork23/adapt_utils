from firedrake import *

from adapt_utils.tracer.stabilisation import *


# TODO: Test something meaningful
mesh = UnitSquareMesh(10, 2)
u = Constant(as_vector((3, 0)))
tau = anisotropic_stabilisation(u, mesh)
print('Done!')
File('plots/tau.pvd').write(tau)

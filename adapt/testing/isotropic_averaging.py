from firedrake import *
# NOTE: We will be able to do this through interpolate soon

mesh = UnitSquareMesh(8, 8)
P0 = FunctionSpace(mesh, 'DG', 0)
P1_ten = TensorFunctionSpace(mesh, 'CG', 1)
M = Function(P1_ten)
phi = Function(P0)
phi.interpolate(...)

i = TestFunction(P0) * Constant(mesh.num_cells())  # indicator function
area = interpolate(assemble(i*dx), P0)  # elemental areas

# for each vertex v:
#    * s = 0
#    * w = 0
#    for each adjacent element K:
#        * s += area[K]    (or should it just be += 1?)
#        * w += area[K] * phi[K]
#    * M[v] = w / s

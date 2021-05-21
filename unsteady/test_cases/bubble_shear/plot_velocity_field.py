from thetis import *

import matplotlib.pyplot as plt

from adapt_utils.unsteady.test_cases.bubble_shear.options import BubbleOptions
from adapt_utils.plotting import *


op = BubbleOptions(n=1)
mesh = op.default_mesh
P1_vec = VectorFunctionSpace(mesh, "CG", 1)
u = Function(P1_vec, name="Fluid velocity")

# Plot to pvd
outfile = File('plots/velocity.pvd')
t = 0.0
tc = Constant(0.0)
while t < op.end_time - 1.0e-05:
    print("t = {:.4f}".format(t))
    tc.assign(t)
    u.interpolate(op.get_velocity(mesh.coordinates, tc))
    outfile.write(u)
    t += op.dt_per_export*op.dt*50

# Plot to jpg
plot_dir = 'plots'
for i in [0, 65, 131, 196]:
    t = 0.0025*i
    u.interpolate(op.get_velocity(mesh.coordinates, t))
    outfile.write(u)

    fig, axes = plt.subplots()
    tc = tricontourf(u, axes=axes, cmap='coolwarm', levels=50)
    cb = fig.colorbar(tc, ax=axes)
    savefig("velocity_{:.2f}".format(t), plot_dir, extensions=["jpg"])

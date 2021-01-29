from firedrake import *
from adapt_utils.mesh import plot_quality
from adapt_utils.plotting import *


mesh = Mesh('circle.msh')
fig, axes = plot_quality(mesh)
axes.set_xlim([-0.01, 1.01])
axes.set_ylim([-0.01, 1.01])
savefig("mesh_quality", "plots", extensions=["pdf"])

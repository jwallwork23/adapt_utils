from firedrake import *
import matplotlib.pyplot as plt
from adapt_utils.mesh import plot_aspect_ratio
from adapt_utils.plotting import *


mesh = Mesh('resources/meshes/channel_0_0.msh')
fig, axes = plt.subplots(figsize=(12, 5))
fig, axes = plot_aspect_ratio(mesh, fig=fig, axes=axes)
axes.set_xlim([-0.1, 1200.1])
axes.set_ylim([-0.1, 500.1])
savefig("aspect_ratio_aligned", "plots", extensions=["pdf"])
plt.close()

mesh = Mesh('resources/meshes/channel_0_1.msh')
fig, axes = plt.subplots(figsize=(12, 5))
fig, axes = plot_aspect_ratio(mesh, fig=fig, axes=axes)
axes.set_xlim([-0.1, 1200.1])
axes.set_ylim([-0.1, 500.1])
savefig("aspect_ratio_offset", "plots", extensions=["pdf"])
plt.close()

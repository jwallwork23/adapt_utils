from thetis import *

import os
import matplotlib.pyplot as plt

from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

op = SpaceshipOptions()
swp = AdaptiveTurbineProblem(op)

# Plot mesh
ax = axes[0]
triplot(swp.meshes[0], axes=ax)
ax.set_title("Mesh")
ax.legend()
ax.axis(False)

# Plot bathymetry
ax = axes[1]
eps = 1.0e-03
cbar_range = np.linspace(4.5 - eps, 25.5 + eps, 11)
fig.colorbar(tricontourf(swp.bathymetry[0], axes=ax, levels=cbar_range, cmap='coolwarm'), ax=ax)
ax.set_title("Bathymetry [m]")
ax.axis(False)
axes[0].set_ylim(ax.get_ylim())

# Save to file
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_and_bathymetry", ext])))

from thetis import *

import os
import matplotlib.pyplot as plt

from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


fig, axes = plt.subplots(ncols=2, figsize=(11, 4))

op = SpaceshipOptions()
swp = AdaptiveTurbineProblem(op)

# Plot mesh
triplot(swp.meshes[0], axes=axes[0])
axes[0].set_title("Mesh")
axes[0].legend()

# Plot bathymetry
ax = axes[1]
eps = 1.0e-03
cbar_range = np.linspace(4.5 - eps, 25.5 + eps, 11)
fig.colorbar(tricontourf(swp.bathymetry[0], axes=ax, levels=cbar_range, cmap='coolwarm'), ax=ax)
axes[1].set_title("Bathmetry [m]")

# Save to file
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
plt.savefig(os.path.join(di, 'mesh_and_bathymetry.png'))

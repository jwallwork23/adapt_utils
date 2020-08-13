from thetis import *

import matplotlib.pyplot as plt
import os

from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


plt.rc('text', usetex=True)

# Create parameter and solver objects
op = SpaceshipOptions()
swp = AdaptiveTurbineProblem(op)

# Plot mesh
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
ax = axes[0]
triplot(swp.meshes[0], axes=ax, interior_kw={'linewidth': 0.2})
ax.legend()
ax.axis(False)

# Plot bathymetry
ax = axes[1]
eps = 1.0e-02
b = swp.bathymetry[0]
cbar_range = np.linspace(4.5 - eps, 25.5 + eps, 50)
cbar = fig.colorbar(tricontourf(b, axes=ax, levels=cbar_range, cmap='coolwarm'), ax=ax)
cbar.set_ticks(np.linspace(4.5, 25.5, 11))
cbar.set_label(r"Bathymetry $[m]$")
ax.axis(False)
ylim = 1.05*np.array(ax.get_ylim())
axes[0].set_ylim(ylim)
ax.set_ylim(ylim)

# Save to file
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_and_bathymetry", ext])))

# Plot linear and exponential sponges
for sponge_type in ('linear', 'exponential'):
    fig, axes = plt.subplots(figsize=(5, 4))
    op.viscosity_sponge_type = sponge_type
    nu = op.set_viscosity(swp.P1[0])
    cbar_range = np.linspace(5.0 - eps, 100.0 + eps, 50)
    cbar = fig.colorbar(tricontourf(nu, levels=cbar_range, axes=axes, cmap='coolwarm'), ax=axes)
    cbar.set_ticks(np.linspace(5.0, 100.0, 20))
    cbar.set_label(r"Kinematic viscosity $[m^2\,s^{-1}]$")
    axes.axis(False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(di, ".".join([sponge_type + "_sponge_viscosity", ext])))

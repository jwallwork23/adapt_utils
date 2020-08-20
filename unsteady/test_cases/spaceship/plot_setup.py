from thetis import *

import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import os

from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


# --- Set parameters

kwargs = {
    "interior_kw": {
        "linewidth": 0.1,
    },
    "boundary_kw": {
        "color": "k",
    },
}
font = {
    "family": "DejaVu Sans",
    "size": 18,
}
plt.rc("font", **font)
plt.rc("text", usetex=True)
patch_kwargs = {
    "facecolor": "none",
    "linewidth": 2,
}
op = SpaceshipOptions()
L = 1.05*op.domain_length
W = 1.05*op.domain_width
swp = AdaptiveTurbineProblem(op)

# --- Plot mesh

# Whole mesh
fig, axes = plt.subplots(figsize=(5, 4))
triplot(swp.meshes[0], axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_xticks(np.linspace(-20000, 20000, 3))
axes.set_yticks(np.linspace(-20000, 20000, 5))

# Annotate turbines
for i, loc in enumerate(op.region_of_interest):
    patch_kwargs["edgecolor"] = "C{:d}".format(i // 3)
    centre = (loc[0]-loc[2]/2, loc[1]-loc[3]/2)
    axes.add_patch(ptch.Rectangle(centre, loc[2], loc[3], **patch_kwargs))

# Save
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh", ext])))

# Zoom of turbine region
axes.set_xlim([-L/12, L/3])
axes.set_ylim([-W/6, W/6])
axes.set_xticks(np.linspace(0, 20000, 3))
axes.set_yticks(np.linspace(-10000, 10000, 5))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_zoom", ext])))

# Zoom again
axes.set_xlim([5000, 10000])
axes.set_ylim([-2500, 2500])
axes.set_xticks(np.linspace(5000, 10000, 3))
axes.set_yticks(np.linspace(-2500, 2500, 5))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_zoom_again", ext])))

# --- Plot bathymetry

fig, axes = plt.subplots(figsize=(5, 4))
eps = 1.0e-02
b = swp.bathymetry[0]
cbar_range = np.linspace(4.5 - eps, 25.5 + eps, 50)
cbar = fig.colorbar(tricontourf(b, axes=axes, levels=cbar_range, cmap='coolwarm_r'), ax=axes)
cbar.set_ticks(np.linspace(4.5, 25.5, 5))
cbar.set_label(r"Bathymetry $[m]$")
axes.axis(False)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["bathymetry", ext])))

# --- Plot linear and exponential sponges

ticks = np.linspace(0.0, op.max_viscosity, 5)
ticks[0] = op.base_viscosity
for sponge_type in ('linear', 'exponential'):
    fig, axes = plt.subplots(figsize=(5, 4))
    op.viscosity_sponge_type = sponge_type
    nu = op.set_viscosity(swp.P1[0])
    cbar_range = np.linspace(op.base_viscosity - eps, op.max_viscosity + eps, 50)
    cbar = fig.colorbar(tricontourf(nu, levels=cbar_range, axes=axes, cmap='coolwarm'), ax=axes)
    cbar.set_ticks(ticks)
    cbar.set_label(r"Kinematic viscosity $[m^2\,s^{-1}]$")
    axes.axis(False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(di, ".".join([sponge_type + "_sponge_viscosity", ext])))

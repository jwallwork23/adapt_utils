from thetis import *

import os
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


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
    "family" : "DejaVu Sans",
    "size"   : 24,
}
plt.rc("font", **font)
plt.rc("text", usetex=True)
patch_kwargs = {
    "facecolor": "none",
    "linewidth": 2,
}
op = TurbineArrayOptions()
L = op.domain_length
W = op.domain_width


# --- Plot whole mesh

fig, axes = plt.subplots(figsize=(12, 6))
triplot(op.default_mesh, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()

# Annotate turbines
for i, loc in enumerate(op.region_of_interest):
    patch_kwargs["edgecolor"] = "C{:d}".format(i // 3)
    centre = (loc[0]-loc[2]/2, loc[1]-loc[3]/2)
    axes.add_patch(ptch.Rectangle(centre, loc[2], loc[3], **patch_kwargs))

# Save
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh", ext])))


# --- Zoom in on array region

z = 6
axes.set_xlim([-L/z, L/z])
axes.set_ylim([-W/z, W/z])
axes.set_yticks(np.linspace(-150, 150, 7))

# Save
for ext in ("png", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_zoom", ext])))

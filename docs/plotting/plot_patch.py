from firedrake import *

import matplotlib.pyplot as plt

from adapt_utils.mesh import get_patch, make_consistent
from adapt_utils.plotting import *


fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
mesh = UnitSquareMesh(4, 4)
plex, offset, coordinates = make_consistent(mesh)

# Interior node
vvv = 44
patch = get_patch(vvv, plex=plex, coordinates=coordinates)
elements = set(patch['elements'].keys())
axes[0].set_title("Interior")
triplot(mesh, axes=axes[0], boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2'
    marker = 'o' if v == vvv else 'x'
    axes[0].plot(*coordinates(v), marker, color=colour)
for k in elements:
    colour = 'C1'
    axes[0].plot(*patch['elements'][k]['centroid'], '^', color=colour)
axes[0].axis(False)

# Boundary node
vvv = 46
patch = get_patch(vvv, plex=plex, coordinates=coordinates)
elements = set(patch['elements'].keys())
orig_elements = set(patch['elements'].keys())
for v in patch['vertices']:
    if len(plex.getSupport(v)) == 6:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=elements)
        break
elements = set(patch['elements'].keys())
axes[1].set_title("Boundary")
triplot(mesh, axes=axes[1], boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2'
    marker = 'o' if v == vvv else 'x'
    axes[1].plot(*coordinates(v), marker, color=colour)
for k in elements:
    colour = 'C1' if k in orig_elements else 'C5'
    axes[1].plot(*patch['elements'][k]['centroid'], '^', color=colour)
axes[1].axis(False)

# Corner node
vvv = 32
patch = get_patch(vvv, plex=plex, coordinates=coordinates)
elements = set(patch['elements'].keys())
orig_elements = set(patch['elements'].keys())
for v in patch['vertices']:
    if len(plex.getSupport(v)) == 4:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=elements)
        break
elements = set(patch['elements'].keys())
for v in patch['vertices']:
    if len(plex.getSupport(v)) == 6:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=elements)
        break
elements = set(patch['elements'].keys())
axes[2].set_title("Corner")
triplot(mesh, axes=axes[2], boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2'
    marker = 'o' if v == vvv else 'x'
    axes[2].plot(*coordinates(v), marker, color=colour)
for k in elements:
    colour = 'C1' if k in orig_elements else 'C5'
    axes[2].plot(*patch['elements'][k]['centroid'], '^', color=colour)
axes[2].axis(False)

plt.show()

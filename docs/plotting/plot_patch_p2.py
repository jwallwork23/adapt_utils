from firedrake import *

import matplotlib.pyplot as plt

from adapt_utils.mesh import get_patch, make_consistent
from adapt_utils.plotting import *


mesh = UnitSquareMesh(4, 4)
bnodes = DirichletBC(FunctionSpace(mesh, "CG", 1), 0, 'on_boundary').nodes
plex, offset, coordinates = make_consistent(mesh)

# Interior node
fig, axes = plt.subplots(figsize=(5, 5))
vvv = 44
patch = get_patch(vvv, plex=plex, coordinates=coordinates, midfacets=True)
facets = set(patch['facets'].keys())
triplot(mesh, axes=axes, boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2'
    marker = 'o' if v == vvv else 'x'
    axes.plot(*coordinates(v), marker, color=colour, markersize=10)
for e in facets:
    colour = 'C6'
    axes.plot(*patch['facets'][e]['midfacet'], 's', color=colour)
axes.axis(False)
savefig('quadratic_patch_interior', 'outputs', extensions=['pdf'])

# Boundary node
fig, axes = plt.subplots(figsize=(5, 5))
vvv = 46
patch = get_patch(vvv, plex=plex, coordinates=coordinates, midfacets=True)
facets = set(patch['facets'].keys())
orig_facets = set(patch['facets'].keys())
orig_vertices = patch['vertices']
for v in patch['vertices']:
    if offset(v) not in bnodes:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=patch, midfacets=True)
        break
facets = set(patch['facets'].keys())
triplot(mesh, axes=axes, boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2' if v in orig_vertices else 'C3'
    marker = 'o' if v == vvv else 'x'
    axes.plot(*coordinates(v), marker, color=colour, markersize=10)
for e in facets:
    colour = 'C6' if e in orig_facets else 'C8'
    axes.plot(*patch['facets'][e]['midfacet'], 's', color=colour)
axes.axis(False)
savefig('quadratic_patch_boundary', 'outputs', extensions=['pdf'])

# Corner node
fig, axes = plt.subplots(figsize=(5, 5))
vvv = 32
patch = get_patch(vvv, plex=plex, coordinates=coordinates, midfacets=True)
facets = set(patch['facets'].keys())
orig_facets = set(patch['facets'].keys())
orig_vertices = patch['vertices']
for v in patch['vertices']:
    if v != vvv and offset(v) in bnodes:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=patch, midfacets=True)
        break
facets = set(patch['facets'].keys())
for v in patch['vertices']:
    if offset(v) not in bnodes:
        patch = get_patch(v, plex=plex, coordinates=coordinates, extend=patch, midfacets=True)
        break
facets = set(patch['facets'].keys())
triplot(mesh, axes=axes, boundary_kw={'color': 'k'})
for v in patch['vertices']:
    colour = 'C4' if v == vvv else 'C2' if v in orig_vertices else 'C3'
    marker = 'o' if v == vvv else 'x'
    axes.plot(*coordinates(v), marker, color=colour, markersize=10)
for e in facets:
    colour = 'C6' if e in orig_facets else 'C8'
    axes.plot(*patch['facets'][e]['midfacet'], 's', color=colour)
axes.axis(False)
savefig('quadratic_patch_corner', 'outputs', extensions=['pdf'])

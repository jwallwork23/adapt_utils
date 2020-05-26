"""
Post-process the qmesh generated meshes using Pragmatic and any -dm_plex command line parameters of
choice.
"""
import firedrake
import thetis

import math
import matplotlib.pyplot as plt

from adapt_utils.adapt.metric import cell_size_metric
from adapt_utils.case_studies.tohoku.options import TohokuOptions


for level in range(4):
    op = TohokuOptions(level=level, h_min=1e3, h_max=1e6)
    mesh = op.default_mesh

    P0 = firedrake.FunctionSpace(mesh, "DG", 0)
    h = firedrake.interpolate(firedrake.CellSize(mesh), P0)
    P1 = firedrake.FunctionSpace(mesh, "CG", 1)
    M = cell_size_metric(mesh, op=op)

    newmesh = firedrake.adapt(mesh, M)

    print("\n" + 40*"=" + "\n           before     after\n" + 40*"=")
    print("#elements: {:8d}     {:8d}".format(mesh.num_cells(), newmesh.num_cells()))
    print("#vertices: {:8d}     {:8d}".format(mesh.num_vertices(), newmesh.num_vertices()))

    P0 = firedrake.FunctionSpace(newmesh, "DG", 0)
    h_new = firedrake.interpolate(firedrake.CellSize(newmesh), P0)
    with h.dat.vec_ro as v_old:
        with h_new.dat.vec_ro as v_new:
            print("min size : {:8.2e}     {:8.2e}".format(v_old.min()[1], v_new.min()[1]))
            print("max size : {:8.2e}     {:8.2e}".format(v_old.max()[1], v_new.max()[1]))
    print("min angle: {:8.2f}     {:8.2f}".format(
        180/math.pi*thetis.get_minimum_angles_2d(mesh).vector().gather().min(),
        180/math.pi*thetis.get_minimum_angles_2d(newmesh).vector().gather().min()
    ))

    fig, ax = plt.subplots(figsize=(7, 8))
    firedrake.triplot(mesh, axes=ax);
    ax.legend();
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);
    ax.axis('off');
    plt.savefig("Tohoku{:d}.pdf".format(level))

    fig, ax = plt.subplots(figsize=(7, 8))
    firedrake.triplot(mesh, axes=ax);
    ax.set_xlim([400e3, 700e3]);
    ax.set_ylim([4050e3, 4450e3]);
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);
    ax.axis('off');
    plt.savefig("Tohoku{:d}_zoom.pdf".format(level))

    fig, ax = plt.subplots(figsize=(7, 8))
    firedrake.triplot(newmesh, axes=ax);
    ax.legend();
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);
    ax.axis('off');
    plt.savefig("Tohoku{:d}_postproc.pdf".format(level))

    fig, ax = plt.subplots(figsize=(7, 8))
    firedrake.triplot(newmesh, axes=ax);
    ax.set_xlim([400e3, 700e3]);
    ax.set_ylim([4050e3, 4450e3]);
    ax.get_xaxis().set_visible(False);
    ax.get_yaxis().set_visible(False);
    ax.axis('off');
    plt.savefig("Tohoku{:d}_postproc_zoom.pdf".format(level))

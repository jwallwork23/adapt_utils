from firedrake import *

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

from adapt_utils.steady.test_cases.turbine_array.options import *
from adapt_utils.steady.swe.turbine.solver import *
from adapt_utils.plotting import *


plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation approach (default fixed_mesh)")
parser.add_argument('-target', help="Target complexity for adaptive approaches (default 3200)")
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh (default 0)")
parser.add_argument('-adapt_field', help="Field(s) for adaptation (default all_int)")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.
    (Default 0)""")
parser.add_argument('-debug', help="Toggle debugging mode (default False)")
parser.add_argument('-save_mesh', help="Save DMPlex to HDF5 (default False)")
parser.add_argument('-save_plot', help="Save plots of mesh and initial fluid speed (default False)")
args = parser.parse_args()
save_mesh = bool(args.save_mesh or False)

kwargs = {
    'approach': args.approach or 'fixed_mesh',
    'offset': int(args.offset or 0),
    'plot_pvd': True,
    'debug': bool(args.debug or 0),

    # Adaptation parameters
    'target': float(args.target or 3200.0),
    'adapt_field': args.adapt_field or 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,  # i.e. infinity norm
    'h_max': 500.0,

    # Optimisation parameters
    'element_rtol': 0.001,
    'max_adapt': 35,

}
level = int(args.level or 0)
save_plot = bool(args.save_plot or False)
op = TurbineArrayOptions(**kwargs)
op.set_all_rtols(op.element_rtol)
if op.approach != 'fixed_mesh':
    level = 1
tp = SteadyTurbineProblem(op, discrete_adjoint=True, levels=level)

# Farm geometry
loc = op.region_of_interest
D = op.turbine_diameter
centre_t1 = (loc[0][0]-D/2, loc[0][1]-D/2)
centre_t2 = (loc[1][0]-D/2, loc[1][1]-D/2)
patch_kwargs = {'facecolor': 'none', 'edgecolor': 'b', 'linewidth': 2}

if tp.op.approach == 'fixed_mesh':  # TODO: Use 'uniform' approach?

    # Plot initial mesh
    if save_plot:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        triplot(tp.mesh, axes=ax)
        ax.set_xlim([0.0, op.domain_length])
        ax.set_ylim([0.0, op.domain_width])
        ax.add_patch(ptch.Rectangle(centre_t1, D, D, **patch_kwargs))
        ax.add_patch(ptch.Rectangle(centre_t2, D, D, **patch_kwargs))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(24)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(24)
        plt.savefig('screenshots/inital_mesh_offset{:d}_elem{:d}.pdf'.format(op.offset, tp.mesh.num_cells()), bbox_inches='tight')

    # Solve problem in enriched space
    for i in range(level):
        tp = tp.tp_enriched
    tp.solve()
    tp.op.print_debug("QoI: {:.4e}kW".format(tp.quantity_of_interest()/1000))  # TODO: MegaWatts?

    # Plot fluid speed
    if save_plot:
        u = tp.solution.split()[0]
        spd = interpolate(sqrt(dot(u, u)), tp.P1)
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        fig.colorbar(tricontourf(spd, axes=ax, cmap='coolwarm', vmin=3.5, vmax=5.2), ax=ax)
        ax.set_xlim([0.0, op.domain_length])
        ax.set_ylim([0.0, op.domain_width])
        plt.savefig('screenshots/fluid_speed_offset{:d}_elem{:d}.pdf'.format(op.offset, tp.mesh.num_cells()), bbox_inches='tight')
else:
    tp.adaptation_loop()
    if save_mesh:
        tp.save_meshes('{:s}_{:d}.h5'.format(op.approach, op.offset), op.di)

    # Setup figures
    if save_plot:
        fig = plt.figure(figsize=(24, 5))
        ax_main = fig.add_subplot(121)
        ax_zoom = fig.add_subplot(122)

        # Plot mesh and annotate with turbine footprint
        for ax in (ax_main, ax_zoom):
            triplot(tp.mesh, axes=ax)
            ax.add_patch(ptch.Rectangle(centre_t1, D, D, **patch_kwargs))
            ax.add_patch(ptch.Rectangle(centre_t2, D, D, **patch_kwargs))

        # Magnify turbine region
        ax_main.set_xlim([0.0, op.domain_length])
        ax_main.set_ylim([0.0, op.domain_width])
        ax_zoom.set_xlim(centre_t1[0] - 2*D, centre_t2[0] + 2*D)
        ax_zoom.set_ylim(op.domain_width/2 - 3.5*D, op.domain_width/2 + 3.5*D)
        zoom_effect02(ax_zoom, ax, color='w')  # TODO: Remove and delete in plotting.py
        # TODO: Use http://akuederle.com/matplotlib-zoomed-up-inset instead

        # Save to file
        fname = '{:s}_offset{:d}_target{:d}_elem{:d}'.format(op.approach, op.offset, int(op.target), tp.num_cells[-1])
        plt.savefig('screenshots/{:s}.pdf'.format(fname), bbox_inches='tight')

        # Plot dwr cell residual
        residual = interpolate(abs(tp.indicators['dwr_cell']), tp.indicators['dwr_cell'].function_space())
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        # plot(residual, axes=ax, colorbar={'orientation': 'horizontal'}, locator=ticker.LogLocator(), vmin=3.5, vmax=5.2, shading='gouraud')
        plot(residual, axes=ax, colorbar={'orientation': 'horizontal', 'norm': matplotlib.colors.LogNorm()}, vmin=3.5, vmax=5.2, shading='gouraud')
        ax.set_xlim([0.0, op.domain_length])
        ax.set_ylim([0.0, op.domain_width])
        plt.savefig('screenshots/cell_residual_offset{:d}_elem{:d}.pdf'.format(op.offset, tp.mesh.num_cells()), bbox_inches='tight')

        # Plot dwr flux
        flux = interpolate(abs(tp.indicators['dwr_flux']), tp.indicators['dwr_flux'].function_space())
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        # plot(flux, axes=ax, colorbar={'orientation': 'horizontal'}, locator=matplotlib.ticker.LogLocator(), vmin=3.5, vmax=5.2, shading='gouraud')
        plot(flux, axes=ax, colorbar={'orientation': 'horizontal', 'norm': matplotlib.colors.LogNorm()}, vmin=3.5, vmax=5.2, shading='gouraud')
        ax.set_xlim([0.0, op.domain_length])
        ax.set_ylim([0.0, op.domain_width])
        plt.savefig('screenshots/flux_offset{:d}_elem{:d}.pdf'.format(op.offset, tp.mesh.num_cells()), bbox_inches='tight')

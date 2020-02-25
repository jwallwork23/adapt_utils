import argparse
import firedrake
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

from adapt_utils.test_cases.steady_turbine.options import *
from adapt_utils.swe.turbine.solver import *
from adapt_utils.plotting import *


plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument('-approach', help="Mesh adaptation approach.")
parser.add_argument('-target', help="Target complexity for adaptive approaches.")
parser.add_argument('-level', help="Number of uniform refinements to apply to the initial mesh.")
parser.add_argument('-offset', help="""
    Number of turbine diameters by which to offset turbines in y-direction.
    'Aligned' configuration given by offset=0, 'Offset' configuration given by offset=1.""")
args = parser.parse_args()

kwargs = {
    'approach': args.approach or 'fixed_mesh',
    'offset': int(args.offset or 0),
    'plot_pvd': True,
    'debug': True,

    # Adaptation parameters
    'target': float(args.target or 3200.0),
    'adapt_field': 'all_int',
    'normalisation': 'complexity',
    'convergence_rate': 1,
    'norm_order': None,
    'h_max': 500.0,

    # Optimisation parameters
    'element_rtol': 0.002,  # TODO: use 0.001?
    'num_adapt': 35,

}
level = int(args.level or 4)
op = Steady2TurbineOptions(**kwargs)
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
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    meshplot(tp.mesh, axes=ax)
    # firedrake.plot(tp.mesh, axes=ax)
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
    tp.op.print_debug("QoI: {:.4e}kW".format(tp.quantity_of_interest()/1000))

    # Plot fluid speed
    u = tp.solution.split()[0]
    spd = firedrake.interpolate(firedrake.sqrt(firedrake.dot(u, u)), tp.P1)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    firedrake.plot(spd, axes=ax, colorbar={'orientation': 'horizontal'}, vmin=3.5, vmax=5.2, shading='gouraud')
    ax.set_xlim([0.0, op.domain_length])
    ax.set_ylim([0.0, op.domain_width])
    plt.savefig('screenshots/fluid_speed_offset{:d}_elem{:d}.pdf'.format(op.offset, tp.mesh.num_cells()), bbox_inches='tight')
else:
    tp.adaptation_loop()

    # Setup figures
    fig = plt.figure(figsize=(24, 5))
    ax_main = fig.add_subplot(121)
    ax_zoom = fig.add_subplot(122)

    # Plot mesh and annotate with turbine footprint
    for ax in (ax_main, ax_zoom):
        meshplot(tp.mesh, axes=ax)
        ax.add_patch(ptch.Rectangle(centre_t1, D, D, **patch_kwargs))
        ax.add_patch(ptch.Rectangle(centre_t2, D, D, **patch_kwargs))

    # Magnify turbine region
    ax_main.set_xlim([0.0, op.domain_length])
    ax_main.set_ylim([0.0, op.domain_width])
    ax_zoom.set_xlim(centre_t1[0] - 2*D, centre_t2[0] + 2*D)
    ax_zoom.set_ylim(op.domain_width/2 - 3.5*D, op.domain_width/2 + 3.5*D)
    zoom_effect02(ax_zoom, ax, color='w')

    # Save to file
    fname = '{:s}_offset{:d}_target{:d}_elem{:d}'.format(op.approach, op.offset, int(op.target), tp.num_cells[-1])
    plt.savefig('screenshots/{:s}.pdf'.format(fname), bbox_inches='tight')

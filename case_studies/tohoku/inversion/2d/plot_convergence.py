from thetis import COMM_WORLD, create_directory, print_output

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sys

from adapt_utils.argparse import ArgumentParser
from adapt_utils.case_studies.tohoku.options.radial_options import TohokuRadialBasisOptions
from adapt_utils.plotting import *


# --- Parse arguments

parser = ArgumentParser(
    prog="plot_convergence",
    description="""
        Given tsunami source inversion run output, generate a variety of plots:
          (a) timeseries due to initial guess vs. synthetic data;
          (b) progress of the QoI during the optimisation, along with the approximate gradients;
          (c) timeseries due to converged control parameters vs. synthetic data.
        """,
    adjoint=True,
    plotting=True,
)
parser.add_argument("-level", help="Mesh resolution level")
parser.add_argument("-continuous_timeseries", help="Toggle discrete or continuous timeseries data")
parser.add_argument("-regularisation", help="Parameter for Tikhonov regularisation term")


# --- Set parameters

# Parsed arguments
args = parser.parse_args()
level = int(args.level or 0)
plot = parser.plotting_args()
if not plot.any:
    print_output("Nothing to plot.")
    sys.exit(0)
timeseries = 'timeseries'
if bool(args.continuous_timeseries or False):
    timeseries = '_'.join([timeseries, 'smooth'])

# Do not attempt to plot in parallel
if COMM_WORLD.size > 1 and plot.any:
    print_output(120*'*' + "\nWARNING: Plotting turned off when running in parallel.\n" + 120*'*')
    plot.pdf = plot.png = False

# Setup output directories
dirname = os.path.dirname(__file__)
if args.adjoint is None or args.adjoint not in ('discrete', 'continuous'):
    raise ValueError
di = create_directory(os.path.join(dirname, 'outputs', 'synthetic'))
if args.extension is not None:
    di = '_'.join([di, args.extension])
plot_dir = create_directory(os.path.join(di, 'plots'))
create_directory(os.path.join(plot_dir, args.adjoint))

# Collect initialisation parameters
kwargs = {
    'level': level,
    'save_timeseries': True,

    # Optimisation
    'synthetic': True,
    'qoi_scaling': 1.0,
    'nx': 1,
    'ny': 1,
    'regularisation': float(args.regularisation or 0.0),

    # Misc
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
    'di': di,
}
use_regularisation = not np.isclose(kwargs['regularisation'], 0.0)
op = TohokuRadialBasisOptions(**kwargs)
gauges = list(op.gauges.keys())

# Plotting parameters
fontsize = 22
fontsize_tick = 18
fontsize_legend = 18
kwargs = {'markevery': 5}


# --- Plot optimisation progress

# Load parameter spaces
n = 8
xc = yc = np.linspace(0.5, 7.5, n)
Xc, Yc = np.meshgrid(xc, yc)
func_values = np.load(os.path.join(di, 'parameter_space_{:d}.npy'.format(level)))
if use_regularisation:
    func_values_reg = np.load(os.path.join(di, 'parameter_space_reg_{:d}.npy'.format(level)))

# Fit a cubic
q = scipy.interpolate.interp2d(xc, yc, func_values, kind='cubic')

# Fit quadratic to regularised functional values
if use_regularisation:
    q_reg = scipy.interpolate.interp2d(xc, yc, func_values_reg, kind='cubic')
    raise NotImplementedError  # TODO

# Evaluate interpolant and its gradient
xc0, xcf = 1.5, 5.5
yc0, ycf = 1.5, 7.5
xf = np.linspace(xc0, xcf, int(10*(xcf - xc0)))
yf = np.linspace(yc0, ycf, int(10*(ycf - yc0)))
Xf, Yf = np.meshgrid(xf, yf)
Q = q(xf, yf)
xm = np.linspace(xc0, xcf, int(2*(xcf - xc0)))
ym = np.linspace(yc0, ycf, int(2*(ycf - yc0)))
Xm, Ym = np.meshgrid(xm, ym)
dQx = q(xm, ym, dx=1)
dQy = q(xm, ym, dy=1)

# Plot parameter space
fig, axes = plt.subplots(figsize=(8, 8))
params = {'linewidths': 1, 'linestyles': 'dashed', 'levels': 30, 'cmap': 'Blues_r'}
axes.contour(Xf, Yf, Q, **params)
axes.quiver(Xm, Ym, -dQx, -dQy, color='C0')
axes.set_xlabel(r"First basis function coefficient, $m_1$", fontsize=fontsize)
axes.set_ylabel(r"Second basis function coefficient, $m_2$", fontsize=fontsize)
plt.xticks(fontsize=fontsize_tick)
plt.yticks(fontsize=fontsize_tick)
plt.xlim([xc0, xcf])
plt.ylim([yc0, ycf])
axes.grid()

# Establish quadratic
vandermonde = lambda x, y: np.array([x**2, x*y, y**2, x, y, 1], dtype=object)
A = np.zeros((6, 6))
b = np.zeros(6)
i = 0
for i, (x, y) in enumerate(4 + 3*np.random.rand(6, 2)):
    A[i, :] = vandermonde(x, y)
    b[i] = q(x, y)
coeffs = np.linalg.solve(A, b)
q_poly = lambda x, y: np.dot(coeffs, vandermonde(x, y))
Q_poly = q_poly(Xf, Yf)
# assert np.allclose(np.abs((Q - Q_poly)/Q), 0.0, atol=1.0e-03)  # Check we do indeed have a quadratic

# Find root and plot it
a, b, c, d, e, f = coeffs
A = np.array([[2*a, b],
              [b, 2*c]])
b = np.array([-d, -e])
q_min = np.linalg.solve(A, b)
params = {'markersize': 14, 'color': 'C0', 'label': r'$m^\star = ({:.4f}, {:.4f})$'.format(*q_min)}
axes.plot(*q_min, '*', **params)

# Save parameter space plot
fname = 'parameter_space'
if use_regularisation:
    fname += '_reg'
savefig('{:s}_{:d}'.format(fname, level), plot_dir, extensions=plot.extensions)

# Load trajectory
fname = os.path.join(di, args.adjoint, 'optimisation_progress_{:s}' + '_{:d}.npy'.format(level))
control_values_opt = np.load(fname.format('ctrl', level))
func_values_opt = np.load(fname.format('func', level))
gradient_values_opt = np.load(fname.format('grad', level))
optimised_value = control_values_opt[-1]

# Plot trajectory and computed gradients
params = {'markersize': 8, 'color': 'C1', 'label': 'Optimisation progress'}
control_x = [m[0] for m in control_values_opt]
control_y = [m[1] for m in control_values_opt]
axes.plot(control_x, control_y, '-o', **params)
params = {'linewidth': 3, 'color': 'C2'}
# l = 8
gradient_x = [g[0] for g in gradient_values_opt]
gradient_y = [g[1] for g in gradient_values_opt]
# for i, (mx, my, gx, gy) in enumerate(zip(control_x[:l], control_y[:l], gradient_x[:l], gradient_y[:l])):
for i, (mx, my, gx, gy) in enumerate(zip(control_x, control_y, gradient_x, gradient_y)):
    q = axes.quiver([mx], [my], [-gx], [-gy], **params)
axes.quiverkey(q, 0.55, 1.02, 0.5, 'Computed gradient', labelpos='E', **params)
plt.legend(fontsize=fontsize)
axes.annotate(
    r'$m = ({:.4f}, {:.4f})$'.format(*control_values_opt[-1]), weight='bold',
    xy=(2.8, 2.25), color='C1', fontsize=fontsize,
)

# Save optimisation progress plot
fname = 'optimisation_progress'
if use_regularisation:
    fname += '_reg'
savefig(fname + '_{:d}'.format(level), plot_dir, args.adjoint, extensions=plot.extensions)


# --- Plot timeseries

# Before optimisation
msg = "Cannot plot timeseries for initial guess controls on mesh {:d} because the data don't exist."
N = int(np.ceil(np.sqrt(len(gauges))))
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
for i, gauge in enumerate(gauges):

    # Load data
    fname = os.path.join(di, '{:s}_data_{:d}.npy'.format(gauge, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['data'] = np.load(fname)
    fname = os.path.join(di, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['init'] = np.load(fname)
    data = np.array(op.gauges[gauge]['data'])
    init = np.array(op.gauges[gauge]['init'])
    n = len(data)

    T = np.array(op.gauges[gauge]['times'])/60
    T = np.linspace(T[0], T[-1], n)
    ax = axes[i//N, i % N]
    ax.plot(T, data, '-', **kwargs)
    ax.plot(T, init, '-', label=gauge, **kwargs)
    ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
    if i//N == 3:
        ax.set_xlabel('Time (min)', fontsize=fontsize)
    if i % N == 0:
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    ax.grid()
for i in range(len(gauges), N*N):
    axes[i//N, i % N].axis(False)
savefig('timeseries_{:d}'.format(level), plot_dir, extensions=plot.extensions)

# After optimisation
msg = "Cannot plot timeseries for optimised controls on mesh {:d} because the data don't exist."
fig, axes = plt.subplots(nrows=N, ncols=N, figsize=(14, 12))
plotted = False
for i, gauge in enumerate(gauges):

    # Load data
    fname = os.path.join(op.di, args.adjoint, '{:s}_{:s}_{:d}.npy'.format(gauge, timeseries, level))
    if not os.path.isfile(fname):
        print_output(msg.format(level))
        break
    op.gauges[gauge]['opt'] = np.load(fname)
    data = np.array(op.gauges[gauge]['data'])
    opt = np.array(op.gauges[gauge]['opt'])
    n = len(opt)
    assert len(data) == n

    T = np.array(op.gauges[gauge]['times'])/60
    T = np.linspace(T[0], T[-1], n)
    ax = axes[i//N, i % N]
    ax.plot(T, data, '-', **kwargs)
    ax.plot(T, opt, '-', label=gauge, **kwargs)
    ax.legend(handlelength=0, handletextpad=0, fontsize=fontsize_legend)
    if i//N == 3:
        ax.set_xlabel('Time (min)', fontsize=fontsize)
    if i % N == 0:
        ax.set_ylabel('Elevation (m)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize_tick)
    plt.yticks(fontsize=fontsize_tick)
    ax.set_yticks(ax.get_yticks().tolist())  # Avoid matplotlib error
    ax.set_yticklabels(["{:.1f}".format(tick) for tick in ax.get_yticks()])
    t0 = op.gauges[gauge]["arrival_time"]/60
    tf = op.gauges[gauge]["departure_time"]/60
    ax.set_xlim([t0, tf])
    ax.grid()
    plotted = True
if plotted:
    for i in range(len(gauges), N*N):
        axes[i//N, i % N].axis(False)
    fname = 'timeseries_optimised_{:d}'.format(level)
    savefig(fname, plot_dir, args.adjoint, extensions=plot.extensions)
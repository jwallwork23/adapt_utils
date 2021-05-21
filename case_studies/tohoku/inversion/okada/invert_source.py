import adolc
import matplotlib.pyplot as plt
import numpy as np
import scipy

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.optimisation import taylor_test
from adapt_utils.plotting import *
from adapt_utils.norms import vecnorm


def use_degrees(axes, x=True, y=True):
    if x:
        xlim = axes.get_xlim()
        axes.set_xticks(axes.get_xticks())
        axes.set_xticklabels([r"${:.0f}^\circ$".format(tick) for tick in axes.get_xticks()])
        axes.set_xlim(xlim)
    if y:
        ylim = axes.get_ylim()
        axes.set_yticks(axes.get_yticks())
        axes.set_yticklabels([r"${:.0f}^\circ$".format(tick) for tick in axes.get_yticks()])
        axes.set_ylim(ylim)


def use_percent(axes, x=True, y=True):
    if x:
        xlim = axes.get_xlim()
        axes.set_xticks(axes.get_xticks())
        axes.set_xticklabels([r"{:.0f}\%".format(tick) for tick in axes.get_xticks()])
        axes.set_xlim(xlim)
    if y:
        ylim = axes.get_ylim()
        axes.set_yticks(axes.get_yticks())
        axes.set_yticklabels([r"{:.0f}\%".format(tick) for tick in axes.get_yticks()])
        axes.set_ylim(ylim)


np.random.seed(0)  # make results reproducible


# Initialisation
kwargs = {
    'okada_grid_resolution': 51,
    'okada_grid_lon_min': 140,
    'okada_grid_lon_max': 145,
    'okada_grid_lat_min': 35,
    'okada_grid_lat_max': 41,
    'debug': False,
}
active_controls = ['slip', 'rake']
plotting_kwargs = dict(cmap='coolwarm', levels=50)
fontsize = 30
tick_fontsize = 26
op = TohokuOkadaBasisOptions(**kwargs)
op.active_controls = active_controls
op.create_topography()
N = op.N
X = op.fault.dtopo.X
Y = op.fault.dtopo.Y
eta = op.fault.dtopo.dZ.copy()
m_orig = op.input_vector.copy()
np.save("data/m_orig.npy", m_orig)

# Differentiate the source model
kwargs['control_parameters'] = op.control_parameters
for control in op.active_controls:
    size = np.shape(op.control_parameters[control])
    std = np.std(op.control_parameters[control])
    kwargs['control_parameters'][control] += np.random.normal(loc=0, scale=std/2, size=size)
kwargs['control_parameters']['slip'] = np.abs(kwargs['control_parameters']['slip'])
tape_tag = 0
op = TohokuOkadaBasisOptions(**kwargs)
op._data_to_interpolate = eta
op.active_controls = active_controls
op.create_topography(annotate=True, interpolate=True, tag=tape_tag)
print("QoI = {:.4e}".format(op.J.val))
assert np.isclose(op.J.val, 7.1307e-01, rtol=1.0e-04)  # from previous run
op.J_progress = []
op.dJdm_progress = []


def reduced_functional(m):
    """
    Apply the Okada model by unrolling the tape and compute the QoI.
    """
    J = sum(adolc.zos_forward(tape_tag, m, keep=1))
    op.J_progress.append(J)
    return J


def gradient(m):
    """
    Compute the gradient of the QoI with respect to the input parameters.
    """
    dJdm = adolc.fos_reverse(tape_tag, 1.0)
    op.dJdm_progress.append(vecnorm(dJdm, order=np.Inf))
    return dJdm


J = reduced_functional(op.input_vector)
assert np.isclose(J, op.J.val)
g = gradient(op.input_vector)
assert len(g) == len(op.input_vector)
print("J = {:.4e}  ||dJdm|| = {:.4e}".format(op.J_progress[-1], op.dJdm_progress[-1]))
assert np.isclose(op.dJdm_progress[-1], 4.6331e-03, rtol=1.0e-04)  # from previous run

# Plot optimum and initial guess
eta_pert = op.fault.dtopo.dZ.copy()
eta_pert = op.fault.dtopo.dZ.reshape(N, N)
fig, axes = plt.subplots(figsize=(7, 7))
cbar = fig.colorbar(axes.contourf(X, Y, eta, **plotting_kwargs), ax=axes)
cbar.set_label(r"Elevation [$\mathrm m$]", fontsize=fontsize)
cbar.ax.tick_params(labelsize=tick_fontsize)
axes.set_xlabel("Longitude", fontsize=fontsize)
axes.set_ylabel("Latitude", fontsize=fontsize)
use_degrees(axes)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
savefig("original_source", "plots", extensions=["jpg"])
fig, axes = plt.subplots(figsize=(7, 7))
cbar = fig.colorbar(axes.contourf(X, Y, eta_pert, **plotting_kwargs), ax=axes)
cbar.ax.tick_params(labelsize=tick_fontsize)
cbar.set_label(r"Elevation [$\mathrm m$]", fontsize=fontsize)
axes.set_xlabel("Longitude", fontsize=fontsize)
axes.set_ylabel("Latitude", fontsize=fontsize)
use_degrees(axes)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
savefig("perturbed_source", "plots", extensions=["jpg"])

# Taylor test
taylor_test(reduced_functional, gradient, op.input_vector, verbose=True)


def opt_cb(m):
    """
    Print progress after every successful line search.
    """
    msg = "{:4d}: J = {:.4e}  ||dJdm|| = {:.4e}"
    counter = len(op.J_progress)
    if counter % 100 == 0:
        print(msg.format(counter, op.J_progress[-1], op.dJdm_progress[-1]))


# Inversion
op.J_progress = []
op.dJdm_progress = []
bounds = [bound for subfault in op.subfaults for bound in [(0, np.Inf), (-np.Inf, np.Inf)]]
opt_parameters = {
    'maxiter': 40000,
    'disp': True,
    'pgtol': 1.0e-08,
    'callback': opt_cb,
    'bounds': bounds,
    'fprime': gradient,
}
m_opt, J_opt, out = scipy.optimize.fmin_l_bfgs_b(reduced_functional, op.input_vector, **opt_parameters)
np.save("data/m_opt.npy", m_opt)
np.save("data/J_progress.npy", op.J_progress)
np.save("data/dJdm_progress.npy", op.dJdm_progress)

# Plot progress
fig, axes = plt.subplots()
axes.semilogy(op.J_progress)
axes.set_xlabel("Iteration", fontsize=fontsize)
axes.set_ylabel("Mean square error", fontsize=fontsize)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
axes.grid(True)
savefig("J_progress", "plots", extensions=["pdf"])
fig, axes = plt.subplots()
axes.semilogy(op.dJdm_progress)
axes.set_xlabel("Iteration", fontsize=fontsize)
axes.set_ylabel(r"$\ell_\infty$-norm of gradient", fontsize=fontsize)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
axes.grid(True)
savefig("dJdm_progress", "plots", extensions=["pdf"])

# Compare controls
diff = m_opt - m_orig
print("Mean square error of controls = {:.4e}".format(np.linalg.norm(diff)**2/len(m_opt)))
diff = diff.reshape(op.nx*op.ny, len(op.active_controls))
m_orig = m_orig.reshape(op.nx*op.ny, len(op.active_controls))
m_opt = m_opt.reshape(op.nx*op.ny, len(op.active_controls))
fig, axes = plt.subplots(figsize=(7, 7))
labels = [control.capitalize() for control in op.active_controls]
colours = ['C0', 'C9']
axes.hist(diff, bins=21, histtype='bar', density=True, stacked=True, label=labels, color=colours)
plt.yscale('log')
use_degrees(axes, x=True, y=False)
axes.set_xlabel("Difference in control parameter")
axes.set_ylabel("Number of control parameters")
axes.grid(True)
axes.legend()
savefig("histogram", "plots", extensions=["pdf"])
msg = "Maximum pointwise difference: {:.4f} degree difference in {:s} on subfault {:d}"
loc = np.unravel_index(np.argmax(np.abs(diff), axis=None), diff.shape)
print(msg.format(diff[loc], op.active_controls[loc[1]], loc[0]))
rel_diff = diff.copy()
rel_diff[:, 0] /= np.abs(m_orig[:, 0].max())
rel_diff[:, 1] /= np.abs(m_orig[:, 1].max())
rel_diff *= 100
fig, axes = plt.subplots(figsize=(7, 7))
colours = ['C0', 'C9']
markers = ['o', 'x']
for i, (control, colour, marker) in enumerate(zip(op.active_controls, colours, markers)):
    axes.plot(m_orig[:, 0], rel_diff[:, i], marker, label=control.capitalize(), color=colour)
axes.grid(True)
axes.set_xlabel(r"Target slip [$\mathrm m$]")
axes.set_ylabel("Relative difference")
axes.set_yticks([-40, -20, 0, 20, 40])
use_percent(axes, x=False, y=True)
axes.legend()
savefig("slip_vs_diff", "plots", extensions=["pdf"])
fig, axes = plt.subplots(figsize=(6, 5))
colours = ['C0', 'C9']
markers = ['o', 'x']
for i, (control, colour, marker) in enumerate(zip(op.active_controls, colours, markers)):
    axes.plot(m_orig[:, 1], diff[:, i], marker, label=control.capitalize(), color=colour)
axes.grid(True)
axes.set_xlabel("Target rake")
axes.set_ylabel("Relative difference")
axes.set_xticks([30, 45, 60, 75, 90])
use_percent(axes, x=False, y=True)
use_degrees(axes, x=True, y=False)
savefig("rake_vs_diff", "plots", extensions=["pdf"])

# Check pointwise error
for i, control in enumerate(op.active_controls):
    kwargs['control_parameters'][control] = m_opt[:, i]
op = TohokuOkadaBasisOptions(**kwargs)
op.active_controls = active_controls
op.create_topography()
eta_opt = op.fault.dtopo.dZ.reshape(N, N)
eta_err = np.abs(eta - eta_opt)*1000
fig, axes = plt.subplots(figsize=(7, 7))
cbar = fig.colorbar(axes.contourf(X, Y, eta_opt, **plotting_kwargs), ax=axes)
cbar.ax.tick_params(labelsize=tick_fontsize)
cbar.set_label(r'Elevation [$\mathrm m$]', size=fontsize)
axes.set_xlabel("Longitude", fontsize=fontsize)
axes.set_ylabel("Latitude", fontsize=fontsize)
use_degrees(axes)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
savefig("optimised_source", "plots", extensions=["jpg"])
fig, axes = plt.subplots(figsize=(7, 7))
cbar = fig.colorbar(axes.contourf(X, Y, eta_err, **plotting_kwargs), ax=axes)
cbar.ax.tick_params(labelsize=tick_fontsize)
cbar.set_label(r'Elevation [$\mathrm{mm}$]', size=fontsize)
axes.set_xlabel("Longitude", fontsize=fontsize)
axes.set_ylabel("Latitude", fontsize=fontsize)
use_degrees(axes)
for tick in axes.xaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
for tick in axes.yaxis.get_major_ticks():
    tick.label.set_fontsize(tick_fontsize)
savefig("pointwise_error", "plots", extensions=["jpg"])

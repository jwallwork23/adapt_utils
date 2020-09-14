import adolc
import numpy as np

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver import AdaptiveProblem


plot_png = True
kwargs = {
    'level': 0,
    'family': 'cg-cg',  # So we don't need to project when plotting
    'debug': True,
}

# Set Okada parameters to the default
op = TohokuOkadaBasisOptions(**kwargs)

# Perturb the Okada parameters by some Normal noise
kwargs['control_parameters'] = op.control_parameters
mu = 0
sigma = 5
for control in op.active_controls:
    size = np.shape(op.control_parameters[control])
    kwargs['control_parameters'][control] += np.random.normal(loc=mu, scale=sigma, size=size)
op_opt = TohokuOkadaBasisOptions(**kwargs)
for gauge in op_opt.gauges:
    op_opt.gauges[gauge]["data"] = op.gauges[gauge]["data"]

# Create the dislocation field, annotating to ADOL-C's tape with tag 0
tape_tag = 0
swp = AdaptiveProblem(op_opt, nonlinear=False, print_progress=False)
swp.set_initial_condition(annotate_source=True, tag=tape_tag)

if plot_png:
    from thetis import create_directory, tricontourf
    import os
    import matplotlib.pyplot as plt
    from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm

    # Get corners of zoom region
    lonlat_corners = [(138, 32), (148, 42), (138, 42)]
    utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
    xlim = [utm_corners[0][0], utm_corners[1][0]]
    ylim = [utm_corners[0][1], utm_corners[2][1]]

    # Plot perturbed dislocation field
    fig, axes = plt.subplots(figsize=(6, 6))
    eta = swp.fwd_solutions[0].split()[1].copy(deepcopy=True)
    fig.colorbar(tricontourf(eta, axes=axes, cmap='coolwarm', levels=50), ax=axes)
    axes.axis(False)
    axes.set_title("Perturbed source")
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.tight_layout()
    fpath = create_directory(os.path.dirname(__file__), 'plots')
    savefig('perturbed_field', fpath, extensions=['png'])

# Unroll tape
op_opt.get_input_vector()
op_opt.get_seed_matrices()
F, dFdm = adolc.fov_forward(tape_tag, op_opt.input_vector, op_opt.seed_matrices)
N = op_opt.default_mesh.num_vertices()
F = F.reshape(190, N)
F = np.sum(F, axis=0)
dFdm = dFdm.reshape(190, N, 4)
print(dFdm.shape)

if plot_png:

    # Plot compared fields
    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    X = op.fault.dtopo.X
    Y = op.fault.dtopo.Y
    dZ = op.fault.dtopo.dZ.reshape(N)
    dZ_pert = op_opt.fault.dtopo.dZ.reshape(N)
    fig.colorbar(axes[0].contourf(X, Y, dZ, **plotting_kwargs), ax=axes[0])
    axes[0].set_title("Original source")
    fig.colorbar(axes[1].contourf(X, Y, dZ_pert, **plotting_kwargs), ax=axes[1])
    axes[1].set_title("Perturbed source")
    fig.colorbar(axes[2].contourf(X, Y, F, **plotting_kwargs), ax=axes[2])
    axes[2].set_title("Perturbed source by unrolling tape")

dZ_pert = dZ_pert.reshape(F.shape)
assert np.allclose(dZ_pert, F)

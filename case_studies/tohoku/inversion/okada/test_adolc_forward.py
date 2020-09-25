import adolc
import numpy as np

from adapt_utils.case_studies.tohoku.options.okada_options import TohokuOkadaBasisOptions
from adapt_utils.plotting import *
from adapt_utils.unsteady.solver import AdaptiveProblem


np.random.seed(0)  # make results reproducible

plot_png = True
kwargs = {
    'level': 0,
    'family': 'cg-cg',  # So we don't need to project when plotting

    # Source region
    'okada_grid_lon_min': 140,
    'okada_grid_lon_max': 145,
    'okada_grid_lat_min': 35,
    'okada_grid_lat_max': 41,
}
plotting_kwargs = {
    'cmap': 'coolwarm',
    'levels': 50,
}

# Set Okada parameters to the default
op = TohokuOkadaBasisOptions(**kwargs)
swp = AdaptiveProblem(op)
swp.set_initial_condition(annotate_source=False)

# Perturb the Okada parameters by some Normal noise
kwargs['control_parameters'] = op.control_parameters
mu = 0
sigma = 5
for control in op.active_controls:
    size = np.shape(op.control_parameters[control])
    kwargs['control_parameters'][control] += np.random.normal(loc=mu, scale=sigma, size=size)
op_opt = TohokuOkadaBasisOptions(**kwargs)
N = len(op_opt.indices)

# Create the dislocation field, annotating to ADOL-C's tape with tag 0
tape_tag = 0
swp = AdaptiveProblem(op_opt)
swp.set_initial_condition(annotate_source=True, tag=tape_tag)

# Test forward propagation without differentiation
F = adolc.zos_forward(tape_tag, op_opt.input_vector, keep=0)
N = len(op_opt.indices)
F = F.reshape(190, N)
F = np.sum(F, axis=0)
dZ_pert = op_opt.fault.dtopo.dZ.reshape(N)
assert np.allclose(dZ_pert, F)

# Test forward propagation with differentiation
F, dFdm = adolc.fov_forward(tape_tag, op_opt.input_vector, op_opt.seed_matrices)
F = F.reshape(190, N)
F = np.sum(F, axis=0)
dFdm = dFdm.reshape(190, N, 4)

if plot_png:
    from thetis import create_directory
    import os
    import matplotlib.pyplot as plt
    from adapt_utils.unsteady.swe.tsunami.conversion import lonlat_to_utm

    plt.rc('font', **{'size': 16})

    # Get corners of zoom region
    lonlat_corners = [
        (kwargs['okada_grid_lon_min'], kwargs['okada_grid_lat_min']),
        (kwargs['okada_grid_lon_max'], kwargs['okada_grid_lat_min']),
        (kwargs['okada_grid_lon_min'], kwargs['okada_grid_lat_max']),
    ]
    utm_corners = [lonlat_to_utm(*corner, 54) for corner in lonlat_corners]
    xlim = [utm_corners[0][0], utm_corners[1][0]]
    ylim = [utm_corners[0][1], utm_corners[2][1]]

    # Plot compared fields
    fig, axes = plt.subplots(ncols=3, figsize=(14, 4))
    X = op.fault.dtopo.X
    Y = op.fault.dtopo.Y
    dZ = op.fault.dtopo.dZ.reshape(N)
    dZ_pert = op_opt.fault.dtopo.dZ.reshape(N)
    fig.colorbar(axes[0].tricontourf(X, Y, dZ, **plotting_kwargs), ax=axes[0])
    axes[0].set_title("Original source")
    fig.colorbar(axes[1].tricontourf(X, Y, dZ_pert, **plotting_kwargs), ax=axes[1])
    axes[1].set_title("Perturbed source")
    fig.colorbar(axes[2].tricontourf(X, Y, F, **plotting_kwargs), ax=axes[2])
    axes[2].set_title("Perturbed source by unrolling tape")
    di = create_directory(os.path.join(os.path.dirname(__file__), 'plots', 'test_adolc_forward'))
    savefig('compare', di, extensions=['png'])

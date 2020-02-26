import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-loglog")
args = parser.parse_args()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fontsize = 18

def power2error(x):
    return 100*(x - exact)/exact

def error2power(x):
    return x * exact/100 + exact

loglog = bool(args.loglog)
xlabel = "Degrees of freedom (DOFs)"
ylabel = r"Power output $(\mathrm{kW})$"
ylabel2 = r"Relative error in power output (\%)"
if loglog:
    ylabel = ylabel2
errorline = 0.5

characteristics = {
    'fixed_mesh': {'label': 'Uniform refinement', 'marker': 'o', 'color': 'cornflowerblue'},
    'carpio_isotropic': {'label': 'Isotropic adaptation', 'marker': '^', 'color': 'orange'},
    'carpio': {'label': 'Anisotropic adaptation', 'marker': 'x', 'color': 'g'},
}

for offset in (0, 1):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Read converged QoI value from file
    f = h5py.File('outputs/fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset), 'r')
    exact = np.array(f['qoi'])[-1]
    f.close()

    # Plot convergence curves
    for approach in ('fixed_mesh', 'carpio_isotropic', 'carpio'):
        fname = 'outputs/{:s}/hdf5/qoi_offset_{:d}.h5'.format(approach, offset)
        if os.path.exists(fname):
            f = h5py.File(fname, 'r')
            dofs, qois = np.array(f['dofs']), np.array(f['qoi'])
            if loglog and approach == 'fixed_mesh':
                dofs, qois = np.array(dofs[:-1]), np.array(qois[:-1])
            f.close()
            kwargs = characteristics[approach]
            kwargs.update({'linestyle': '-'})
            if loglog:
                ax.loglog(dofs, power2error(qois), **kwargs)
            else:
                ax.semilogx(dofs, qois, **kwargs)
    plt.grid(True)
    # xlim = [2e+3, 1e+6] if loglog else [4e+3, 4e+6]
    # plt.xlim(xlim)
    if not loglog:
        plt.hlines([exact, (1.0 + errorline/100)*exact], xlim[0], xlim[1], linestyles='dashed', label=r'{:.1f}\% relative error'.format(errorline))

    ytick = "{:.2f}\%" if loglog else "{:.2f}"
    scale = 1.0 if loglog else 1e-3
    # if loglog:
    #     ax.set_ylim([0.01, 6.0])
    yticks = [ytick.format(scale*i) for i in ax.get_yticks().tolist()]
    ax.set_yticklabels(yticks)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=16)

    if not loglog:
        secax = ax.secondary_yaxis('right', functions=(power2error, error2power))
        secax.set_ylabel(ylabel2, fontsize=fontsize)
        yticks = ["{:.2f}\%".format(i) for i in secax.get_yticks().tolist()]
        secax.set_yticklabels(yticks)

    fname = 'outputs/convergence_{:d}'.format(offset)
    if loglog:
        fname = '_'.join([fname, 'loglog'])
    plt.savefig(fname + '.png', bbox_inches='tight')

plt.show()

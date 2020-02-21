import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fontsize = 18

def power2error(x):
    return 100*(x - exact)/exact

def error2power(x):
    return x * exact/100 + exact

xlabel = "Degrees of freedom (DOFs)"
ylabel = r"Power output $(\mathrm{kW})$"
ylabel2 = r"Relative error in power output (\%)"
loglog = False

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
        f = h5py.File('outputs/{:s}/hdf5/qoi_offset_{:d}.h5'.format(approach, offset), 'r')
        dofs, qois = np.array(f['dofs']), np.array(f['qoi'])
        if loglog and approach == 'fixed_mesh':
            dofs, qois = np.array(dofs[:-1]), np.array(qois[:-1])
        f.close()
        kwargs = characteristics[approach]
        if loglog:
            ax.loglog(dofs, qois, linestyle='-', **kwargs)
        else:
            ax.semilogx(dofs, qois, linestyle='-', **kwargs)
    plt.grid(True)
    xlim = ax.get_xlim()
    plt.hlines([exact, 1.006*exact], xlim[0], xlim[1], linestyles='dashed', label=r'0.6\% relative error')
    plt.xlim(xlim)

    yticks = ["{:.2f}".format(1e-3*i) for i in ax.get_yticks().tolist()]
    ax.set_yticklabels(yticks)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    secax = ax.secondary_yaxis('right', functions=(power2error, error2power))
    secax.set_ylabel(ylabel2, fontsize=fontsize)
    yticks = ["{:.2f}\%".format(i) for i in secax.get_yticks().tolist()]
    secax.set_yticklabels(yticks)

    plt.savefig('outputs/convergence_{:d}.png'.format(offset))

plt.show()

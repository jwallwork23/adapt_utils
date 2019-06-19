import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

markers = {'Uniform': '--*',
           'Isotropic': '--^', 'Isotropic (av.)': '--^', 'Isotropic (sup.)': ':v',
           'A posteriori': '--s', 'A posteriori (av.)': '--s', 'A posteriori (sup.)': ':D',
           'A priori': '--x', 'A priori (av.)': '--x', 'A priori (sup.)': ':+'}
colours = {'Uniform': 'b',
           'Isotropic': 'g', 'Isotropic (av.)': 'g', 'Isotropic (sup.)': 'g',
           'A posteriori': 'tab:orange', 'A posteriori (av.)': 'tab:orange', 'A posteriori (sup.)': 'tab:orange',
           'A priori': 'm', 'A priori (av.)': 'm', 'A priori (sup.)': 'm'}

__all__ = ["create_dict", "plot_qoi", "plot_error", "plot_estimate", "plot_effectivity"]


def create_dict(centred=True, second_order=False):
    dat = {}
    dat['Uniform'] = {}
    dat['Uniform']['elements'] = [4000, 16000, 64000, 256000, 1024000]
    if centred:
        dat['Uniform']['qoi'] = [2.0547e-01, 1.6873e-01, 1.6259e-01, 1.6343e-01, 1.6345e-01]
    else:
        dat['Uniform']['qoi'] = [8.9190e-02, 7.2197e-02, 6.9363e-02, 6.9720e-02, 6.9722e-02]
    if not second_order:
        dat['Isotropic'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['A posteriori'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['A priori'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []} 
    else:
        dat['Isotropic (av.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['Isotropic (sup.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['A posteriori (av.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['A posteriori (sup.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []}
        dat['A priori (av.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []} 
        dat['A priori (sup.)'] = {'elements': [], 'qoi': [], 'estimator': [], 'iterations': []} 
    return dat

def plot_qoi(dat, centred=True, title=None, filename=None, filepath='plots', err=1):
    J = 0.16344 if centred else 0.06959
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for approach in dat.keys():
        ax.semilogx(dat[approach]['elements'], dat[approach]['qoi'], markers[approach], color=colours[approach], label=approach)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title is not None:
        plt.title(title)
    plt.xlabel(r'Number of mesh elements', fontsize=12)
    plt.xlim([1e3, 1e6])
    n = 1 if centred else 2
    plt.ylabel(r'Quantity of interest, $J_{:d}(\phi_h)$'.format(n), fontsize=14)
    plt.ylim([J-0.01, J+0.01])
    plt.hlines(J, 700, 1.1e6)
    plt.axhspan(J*(1-err/100), J*(1+err/100), alpha=0.5, color='gray', label=r'$\pm{:.1f}\%$'.format(err))
#     plt.legend(bbox_to_anchor=(0.7, 0.0, 0.5, 0.5));
    plt.legend(fontsize=14)
    plt.grid(True);
    if filename is not None:
        plt.savefig('{:s}/{:s}.pdf'.format(filepath, filename))

def plot_error(dat, centred=True, title=None, filename=None, filepath='plots', err=1):
    J = 0.16344 if centred else 0.06959
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for approach in dat.keys():
        J_err = np.array(dat[approach]['qoi']) - J
        J_err = np.abs(J_err)
        J_err /= np.abs(J)
        dat[approach]['qoi_error'] = J_err
        ax.semilogx(dat[approach]['elements'], J_err, markers[approach], color=colours[approach], label=approach)
#     ax.set_aspect(aspect=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.tick_params(axis='both', which='minor', labelsize=16)
    if title is not None:
        plt.title(title)
    plt.xlabel(r'Number of mesh elements', fontsize=12)
    plt.xlim([700, 1.1e6])
    n = 1 if centred else 2
    plt.ylabel(r'Relative error in QoI, $\frac{|J_%d(\phi_h)-J_%d(\phi)|}{|J_%d(\phi)|}$' % (n,n,n), fontsize=14)
    plt.ylim([-1e-3, 0.05])
    plt.hlines(err/100, 1e2, 1e7, linestyles='dotted', label='{:.1f}\% error'.format(err))
    r = np.arange(0, 0.1, step=0.01)
    plt.yticks(r, [r'{:.1f}\%'.format(100*j) for j in r])  # Set locations and labels
#     plt.legend(bbox_to_anchor=(0.6, 0.5, 0.5, 0.5));
    plt.legend(fontsize=14)
    plt.grid(True)
    if filename is not None:
        plt.savefig('{:s}/{:s}.pdf'.format(filepath, filename))
    
def plot_estimate(dat, title=None, filename=None, filepath='plots', second_order=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for approach in dat.keys():
        if approach  != 'Uniform':
            ax.loglog(dat[approach]['elements'], dat[approach]['estimator'], markers[approach], color=colours[approach], label=approach)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title is not None:
        plt.title(title)
    plt.xlabel(r'Number of mesh elements', fontsize=12)
    plt.xlim([700, 1.1e6])
#     plt.ylim([1e-6, 1e-2])
#     plt.ylabel(r'Error estimator, $\eta$', fontsize=14)
    if second_order:
        plt.ylabel(r'Dual Weighted Residual,\\ $\frac12\rho(\phi_h,\phi^*-\phi^*_h)+\frac12\rho^*(\phi_h^*,\phi-\phi_h)$', fontsize=14)
    else:
        plt.ylabel(r'Dual Weighted Residual, $\rho(\phi_h,\phi^*-\phi^*_h)$', fontsize=14)
        plt.tight_layout()
#     plt.legend(bbox_to_anchor=(0.6, 0.0, 0.5, 0.5));
    plt.legend(fontsize=14)
    plt.grid(True)
    if filename is not None:
        plt.savefig('{:s}/{:s}.pdf'.format(filepath, filename))
    
def plot_effectivity(dat, centred=True, title=None, filename=None, filepath='plots'):
    J = 0.16344 if centred else 0.06959
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for approach in dat.keys():
        if approach != 'Uniform':
            estimator = np.array(dat[approach]['estimator'])
            J_err = np.array(dat[approach]['qoi']) - J
            effectivity = np.abs(J_err/estimator)
            ax.loglog(dat[approach]['elements'], effectivity, markers[approach], color=colours[approach], label=approach)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if title is not None:
        plt.title(title)
    plt.xlabel(r'Number of mesh elements', fontsize=12)
    plt.xlim([700, 1.1e6])
    plt.ylabel(r'Effectivity index, $I_{eff}$', fontsize=14)
#     plt.legend(bbox_to_anchor=(0.6, 0.0, 0.5, 0.5));
    plt.legend(fontsize=14)
    plt.grid(True)
    if filename is not None:
        plt.savefig('{:s}/{:s}.pdf'.format(filepath, filename))

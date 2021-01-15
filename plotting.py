import matplotlib
import matplotlib.pyplot as plt
import os


__all__ = ["savefig"]


# Fonts
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 24


def savefig(fname, fpath=None, extensions=['pdf', 'png'], tight=True, **kwargs):
    """
    Save current matplotlib.pyplot figure to file.

    :arg fname: the name to be given to the file.
    :kwarg path: (optional) path to file.
    :kwarg extensions: a list of strings corresponding to the file extensions to be used.
    """
    if tight:
        plt.tight_layout()
    if fpath is not None:
        fname = os.path.join(fpath, fname)
    for extension in extensions:
        plt.savefig('.'.join([fname, extension]), **kwargs)
    if len(extensions) == 0:
        plt.show()

import firedrake
from thetis import create_directory

import os
import scipy.interpolate as si
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from adapt_utils.tsunami.options import TohokuOptions


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

op = TohokuOptions(utm=False)
lon, lat, elev = op.read_bathymetry_file(km=True)
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)

ax1 = axes.flat[0]
cs = ax1.contourf(lon, lat, elev, 50, vmin=-9, vmax=2, cmap=matplotlib.cm.coolwarm)
ax1.set_xlabel("Degrees longitude")
ax1.set_ylabel("Degrees latitude")
ax1.set_title("Original data")

ax2 = axes.flat[1]
firedrake.plot(op.bathymetry, cmap=matplotlib.cm.coolwarm, axes=ax2)
ax2.set_xlabel("Degrees longitude")
ax2.set_ylabel("Degrees latitude")
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_title("Interpolated data")

# TODO: Plot on an anisotropic mesh

cb = fig.colorbar(cs, orientation='horizontal', ax=axes.ravel().tolist(), pad=0.2)
cb.set_label("Bathymetry $[\mathrm k\mathrm m]$")
di = create_directory('outputs/tohoku')
plt.savefig(os.path.join(di, 'bathymetry.pdf'))
plt.show()

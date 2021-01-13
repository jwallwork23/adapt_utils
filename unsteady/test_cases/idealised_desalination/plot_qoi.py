import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import create_directory, index_string
from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.idealised_desalination.options import *


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("-approach", help="Mesh adaptation approach")
parser.add_argument("-level", help="Mesh resolution level in turbine region")
parser.add_argument('-num_meshes', help="Number of meshes in sequence")
parser.add_argument("-debug", help="Print all debugging statements")
parser.add_argument("-debug_mode", help="Choose debugging mode from 'basic' and 'full'")

args = parser.parse_args()


# --- Set parameters

approach = args.approach or 'fixed_mesh'
level = int(args.level or 0)
kwargs = {
    'approach': approach,
    'level': level,
    'num_meshes': int(args.num_meshes or 1),
    'plot_pvd': True,
    'debug': bool(args.debug or False),
    'debug_mode': args.debug_mode or 'basic',
}
op = IdealisedDesalinationOutfallOptions(**kwargs)
index_str = index_string(op.num_meshes)
plt.rc('font', **{'size': 16})


# --- Extract data

timeseries = np.array([])
for i in range(op.num_meshes):
    fname = os.path.join(op.di, "inlet_salinity_diff_{:s}.npy".format(index_string(i)))
    if not os.path.exists(fname):
        raise IOError("Need to run the model in order to get QoI timeseries.")
    timeseries = np.append(timeseries, np.load(fname))


# --- Plot

fig, axes = plt.subplots(figsize=(8, 5))
time_seconds = np.linspace(0.0, op.end_time, len(timeseries))
time_hours = time_seconds/3600
axes.plot(time_hours, timeseries)
axes.set_xlabel(r"Time [$\mathrm h$]")
axes.set_ylabel(r"Inlet salinity difference [$\mathrm{g\,L}^{-1}$]")
axes.set_xlim([0, op.end_time/3600])
axes.grid(True)
non_dimensionalise = lambda time: 3600*time/op.T_tide
dimensionalise = lambda time: 3600*time*op.T_tide
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")
fname = approach
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), "plots"))
savefig('_'.join([fname, "array_power_output", index_str]), plot_dir, extensions=['pdf', 'png'])

import argparse
import numpy as np
import os
import vtk

from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("dxfarm", help="Mesh resolution within turbine farm region (assumed integer).")
args = parser.parse_args()

dxfarm = int(args.dxfarm)

# Get filepath and filename pattern
data_dir = os.path.join(os.path.dirname(__file__), 'data')
fpath = os.path.join(data_dir, 'ramp_3.855cycle_nu0.001_ReMax1000_dxfarm{:d}'.format(dxfarm))
if not os.path.exists(fpath):
    raise IOError("Filepath {:s} does not exist!".format(fpath))
fname = os.path.join(fpath, 'vorticity_{:d}.vtu')

# Create parameters object
op = TurbineArrayOptions(1)
num_timesteps = int(op.T_ramp/op.dt/op.dt_per_export) + 1

# Create an XML reader
reader = vtk.vtkXMLUnstructuredGridReader()

# Extract data
vorticity_min = []
vorticity_max = []
for i in range(num_timesteps):
    reader.SetFileName(fname.format(i))
    reader.Update()
    output = reader.GetOutput()
    zeta_min, zeta_max = output.GetScalarRange()
    print(i, zeta_min, zeta_max)
    vorticity_min.append(zeta_min)
    vorticity_max.append(zeta_max)
vorticity_min = np.array(vorticity_min)
vorticity_max = np.array(vorticity_max)

# Save to NumPy array
fname = os.path.join(data_dir, "vorticity_{:s}_{:d}.npy")
np.save(fname.format('min', dxfarm), vorticity_min)
np.save(fname.format('max', dxfarm), vorticity_max)

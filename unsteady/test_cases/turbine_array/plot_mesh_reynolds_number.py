from firedrake import *

import matplotlib.pyplot as plt
import numpy as np

from adapt_utils.plotting import *
from adapt_utils.unsteady.test_cases.turbine_array.options import TurbineArrayOptions
from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem


# Setup problem
op = TurbineArrayOptions(1.0, debug=True)
op.base_viscosity = 1
op.min_viscosity = 0.01
op.max_reynolds_number = 1000
swp = AdaptiveTurbineProblem(op, ramp_dir='data/ramp')
# op.spun = True
op.spun = False
swp.set_initial_condition()
u, eta = swp.fwd_solutions[0].split()

# # Plot fluid speed
# fig, axes = plt.subplots(figsize=(12, 6))
# tc = tricontourf(u, axes=axes, levels=50, cmap='coolwarm')
# cbar = fig.colorbar(tc, ax=axes)
# cbar.set_label(r"Fluid speed [$\mathrm{m\,s}^{-1}$]")

# Print / plot viscosity
nu = swp.fields[0].horizontal_viscosity
if isinstance(nu, Constant):
    print("Constant (kinematic) viscosity = {:.4e}".format(nu.values()[0]))
else:
    fig, axes = plt.subplots(figsize=(12, 6))
    levels = np.linspace(0.9*op.min_viscosity, 1.1*op.base_viscosity, 50)
    tc = tricontourf(nu, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes)
    cbar.set_label(r"(Kinematic) viscosity [$\mathrm m^2\,\mathrm s^{-1}$]")
    cbar.set_ticks(np.linspace(0, op.base_viscosity, 5))

# Plot mesh Reynolds number
fig, axes = plt.subplots(figsize=(12, 6))
levels = np.linspace(0, 1.1*op.max_reynolds_number, 50)
tc = swp.plot_mesh_reynolds_number(0, axes=axes, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes)
cbar.set_label("Mesh Reynolds number")
cbar.set_ticks(np.linspace(0, op.max_reynolds_number, 5))
plt.tight_layout()
plt.show()

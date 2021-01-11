import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpltools import annotation
import numpy as np
import pandas as pd

from adapt_utils.plotting import *


# --- Total error

resolutions = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
nx = [int(16*5*res) for res in resolutions]

# Get experimental data
data = pd.read_csv('experimental_data.csv', header=None)
N = len(data)

# Compute total errors
total_err = []
for res in resolutions:
    df = pd.read_csv('fixed_output/bed_trench_output_c_{:.4f}.csv'.format(res))
    total_err.append(sum([(df['bath'][i] - data[0].dropna()[i])**2 for i in range(N)]))

# Plot total error against element count
fig, axes = plt.subplots(figsize=(6, 5))
axes.loglog(nx, total_err, '--x')
axes.yaxis.set_minor_formatter(mticker.ScalarFormatter())
axes.set_xlabel(r"Element count in $x$-direction")
axes.set_ylabel(r"$\ell_2$ error")
axes.grid(True)
axes.grid(b=True, which='minor')
savefig("total_error", "plots", extensions=['pdf', 'png'])


# --- Discretisation error

resolutions = [0.0625, 0.125, 0.25, 0.5, 1, 2]
nx = [int(16*5*res) for res in resolutions]

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
N = len(df_real)

# Compute discretisation errors
disc_err = []
for res in resolutions:
    df = pd.read_csv('fixed_output/bed_trench_output_uni_c_{:.4f}.csv'.format(res))
    disc_err.append(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(N)]))

# Plot discretisation error against element count
fig, axes = plt.subplots(figsize=(6, 5))
axes.loglog(nx, disc_err, '--x')
axes.set_xlabel(r"Element count in $x$-direction")
axes.set_ylabel(r"$\ell_2$ error")

# Add slope markers
annotation.slope_marker((20, 1.0e-02), -3, ax=axes, size_frac=0.2)
annotation.slope_marker((14, 1.0e-05), -4, invert=True, ax=axes, size_frac=0.2)

axes.grid(True)
axes.grid(b=True, which='minor')
plt.tight_layout()
plt.show()
savefig("discretisation_error", "plots", extensions=['pdf', 'png'])

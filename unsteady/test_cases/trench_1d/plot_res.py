import matplotlib.pyplot as plt
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
nrm = np.sqrt(sum([data[1].dropna()[i]**2 for i in range(N)]))

# Compute total errors
total_err = []
for res in resolutions:
    df = pd.read_csv('fixed_output/bed_trench_output_c_{:.4f}.csv'.format(res))
    total_err.append(100*np.sqrt(sum([(df['bath'][i] - data[1].dropna()[i])**2 for i in range(N)]))/nrm)

# Plot total error against element count
fig, axes = plt.subplots(figsize=(6, 5))
axes.semilogx(nx, total_err, '-o')
axes.set_xlabel(r"Element count in $x$-direction")
axes.set_ylabel(r"Relative $\ell_2$ error")
yticks = [0, 10, 20, 30, 40]
axes.set_yticks(yticks)
axes.set_yticklabels([r"{{{:.0f}}}\%".format(yt) for yt in yticks])
axes.grid(True)
axes.grid(b=True, which='minor')
savefig("total_error", "plots", extensions=['pdf'])


# --- Discretisation error

resolutions = [0.0625, 0.125, 0.25, 0.5, 1, 2]
nx = [int(16*5*res) for res in resolutions]

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
nrm = np.sqrt(np.sum(df_real['bath']**2))
N = len(df_real)

# Compute discretisation errors
disc_err = []
for res in resolutions:
    df = pd.read_csv('fixed_output/bed_trench_output_uni_c_{:.4f}.csv'.format(res))
    disc_err.append(100*np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(N)]))/nrm)

# Plot discretisation error against element count
fig, axes = plt.subplots(figsize=(6, 5))
axes.loglog(nx, disc_err, '-o')
axes.set_xlabel(r"Element count in $x$-direction")
axes.set_ylabel(r"Relative $\ell_2$ error")
yticks = [0.01, 0.1, 1, 10, 100]
axes.set_yticks(yticks)
axes.set_yticklabels([r"0.01\%", r"0.1\%", r"1\%", r"10\%", r"100\%"])

# Add slope markers
annotation.slope_marker((20, 1.0e-03), -2, invert=True, ax=axes, size_frac=0.2)

axes.grid(True)
axes.grid(b=True, which='minor')
savefig("discretisation_error", "plots", extensions=['pdf'])

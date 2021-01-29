import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from adapt_utils.plotting import *


parser = argparse.ArgumentParser()
parser.add_argument("-freq", help="Mesh movement frequency")
parser.add_argument("-rtol", help="Relative tolerance for Monge-Ampere solver")
args = parser.parse_args()

freq = int(args.freq or 40)
rtol = float(args.rtol or 1.0e-03)
resolutions = [0.0625, 0.125, 0.25]
data = {res: {'disc_err': [], 'total_err': []} for res in resolutions}
alphas = np.linspace(0, 15, 16)
colours = ["C{:d}".format(i) for i in range(len(resolutions))]

# Get experimental data
df_exp = pd.read_csv('experimental_data.csv', header=None)
N = len(data)
nrm_exp = np.sqrt(sum([df_exp[1].dropna()[i]**2 for i in range(N)]))

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
nrm = np.sqrt(np.sum(df_real['bath']**2))

# Get fixed mesh data
for res in resolutions:
    df = pd.read_csv('fixed_output/bed_trench_output_uni_c_{:.4f}.csv'.format(res))
    data[res]['fixed_total_err'] = 100*np.sqrt(sum([(df['bath'][i] - df_exp[1].dropna()[i])**2 for i in range(N)]))/nrm_exp
    data[res]['fixed_disc_err'] = 100*np.sqrt(np.sum((df['bath'] - df_real['bath'])**2))/nrm

# Get errors
fname = 'adapt_output/bed_trench_output_uni_s_{:.4f}_{:.1f}_{:.1e}_{:d}.csv'
for res in resolutions:
    for alpha in alphas:
        df = pd.read_csv(fname.format(res, alpha, rtol, freq))
        data[res]['total_err'].append(100*np.sqrt(sum([(df['bath'][i] - df_exp[1].dropna()[i])**2 for i in range(N)]))/nrm_exp)
        data[res]['disc_err'].append(100*np.sqrt(np.sum((df['bath'] - df_real['bath'])**2))/nrm)

# Plot total error against element count
fig, axes = plt.subplots(figsize=(6, 5))
for res, colour in zip(data, colours):
    label = r'$N_x={{{:d}}}$'.format(int(res*80))
    axes.semilogy(alphas, data[res]['total_err'], '-o', label=label, color=colour)
    axes.hlines(y=data[res]['fixed_total_err'], linestyle='--', color=colour, xmin=alphas[0], xmax=alphas[-1])
axes.set_xlabel(r"Monitor function parameter, $\alpha$")
axes.set_ylabel(r"Relative $\ell_2$ error")
axes.set_xlim([alphas[0], alphas[-1]])
yticks = [1, 10, 100]
axes.set_yticks(yticks)
axes.set_yticklabels([r"1\%", r"10\%", r"100\%"])
axes.set_ylim([10, 100])
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.grid(True)
axes.grid(True, which='minor')
axes.legend(bbox_to_anchor=(0.0, 0.9, 1.0, 0.2), fontsize=18, ncol=2)
savefig("total_error_alpha", "plots", extensions=['pdf'])

# Plot discretisation error against element count
fig, axes = plt.subplots(figsize=(6, 5))
for res, colour in zip(data, colours):
    label = r'$N_x={{{:d}}}$'.format(int(res*80))
    axes.semilogy(alphas, data[res]['disc_err'], '-o', label=label, color=colour)
    axes.hlines(y=data[res]['fixed_disc_err'], linestyle='--', color=colour, xmin=alphas[0], xmax=alphas[-1])
axes.set_xlabel(r"Monitor function parameter, $\alpha$")
axes.set_ylabel(r"Relative $\ell_2$ error")
axes.set_xlim([alphas[0], alphas[-1]])
yticks = [1, 10, 100]
axes.set_yticks(yticks)
axes.set_yticklabels([r"1\%", r"10\%", r"100\%"])
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.grid(True)
axes.grid(True, which='minor')
axes.legend(bbox_to_anchor=(0.0, 0.9, 1.0, 0.2), fontsize=18, ncol=2)
savefig("discretisation_error_alpha", "plots", extensions=['pdf'])

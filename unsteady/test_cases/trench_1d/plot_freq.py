import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import numpy as np
import os
import pandas as pd

from adapt_utils.plotting import *


parser = argparse.ArgumentParser()
parser.add_argument("-res", help="Resolution in x-direction")
args = parser.parse_args()

res = float(args.res or 0.5)
alpha = 2.0

di = os.path.join(os.path.dirname(__file__), 'outputs', 'freq', '{:.4f}'.format(res))
freqs = [5, 10, 20, 40, 120, 360, 1080, 2160]
N = len(freqs)
resolutions, alphas, tol, time = [], [], [], []
for freq in freqs:
    with open(os.path.join(di, str(freq)), 'r') as f:
        resolutions.append(float(f.readline().split('=')[-1]))
        alphas.append(float(f.readline().split('=')[-1]))
        tol.append(float(f.readline().split('=')[-1]))
        time.append(float(f.readline().split(':')[-1][:-2]))
assert np.allclose(res*np.ones(N), resolutions)
assert np.allclose(alpha*np.ones(N), alphas)
assert np.allclose(1.0e-04*np.ones(N), tol)

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')

# Get discretisation errors
disc_err = []
for freq in freqs:
    df = pd.read_csv('adapt_output/bed_trench_output_uni_s_{:.4f}_2.0_1.0e-04_{:d}.csv'.format(res, freq))
    disc_err.append(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

# Get fixed mesh data
df_fixed = pd.read_csv('fixed_output/bed_trench_output_uni_c_{:.4f}.csv'.format(res))
fixed_err = np.sqrt(sum([(df_fixed['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
fname = os.path.join(os.path.dirname(__file__), 'outputs', 'res', '{:.4f}'.format(res))
with open(fname, 'r') as f:
    assert np.isclose(res, float(f.readline().split('=')[-1]))
    fixed_time = float(f.readline().split(':')[-1][:-2])

# Plot both error and time against frequency
fig, axes = plt.subplots(figsize=(10, 5))
host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes.set_xticks([])
axes.set_yticks([])
axes = plt.gca()
freqs = 1.0/np.array(freqs)
disc_err = 100*np.array(disc_err)/fixed_err
time = 100*np.array(time)/fixed_time
p1, = host.plot(freqs, disc_err, '--x')
p2, = par1.plot(freqs, time, '--x')
plt.xscale('log')
host.set_xticks(list(freqs))
host.set_xticklabels([r"$\frac1{{{:d}}}$".format(int(f)) for f in 1.0/freqs])
if np.isclose(res, 0.5):
    host.set_yticks([60, 70, 80, 90])
    host.set_ylim([60, 90])
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900])
    par1.set_ylim([100, 900])
elif np.isclose(res, 1.0):
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800])
    par1.set_ylim([200, 800])
host.set_xlabel("Mesh movement frequency")
host.set_ylabel(r"$\ell_2$ error increase ($\%$)")
par1.set_ylabel(r"Time increase ($\%$)")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("freq_{:.4f}_{:.1f}".format(res, alpha), "plots", extensions=['pdf', 'png'])

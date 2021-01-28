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
    fname = 'adapt_output/bed_trench_output_uni_s_{:.4f}_2.0_1.0e-04_{:d}.csv'.format(res, freq)
    disc_err.append(np.sqrt(np.sum((pd.read_csv(fname)['bath'] - df_real['bath'])**2)))

# Get fixed mesh data
fname = 'fixed_output/bed_trench_output_uni_c_{:.4f}.csv'.format(res)
fixed_err = np.sqrt(np.sum((pd.read_csv(fname)['bath'] - df_real['bath'])**2))

# Get fixed mesh time
fname = os.path.join(os.path.dirname(__file__), 'outputs', 'res', '{:.4f}'.format(res))
with open(fname, 'r') as f:
    assert np.isclose(res, float(f.readline().split('=')[-1]))
    fixed_time = float(f.readline().split(':')[-1][:-2])

# Plot both error and time against frequency
fig, axes = plt.subplots(figsize=(8, 5))
host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes.set_xticks([])
axes.set_yticks([])
axes = plt.gca()
freqs = 1.0/np.array(freqs)
disc_err = 100*np.array(disc_err)/fixed_err
time = 100*np.array(time)/fixed_time
p1, = host.plot(freqs, disc_err, '-o', linewidth=2)
p2, = par1.plot(freqs, time, '-o', linewidth=2)
plt.xscale('log')
host.set_xticks(list(freqs))
host.set_xticklabels([r"$\frac1{{{:d}}}$".format(int(f)) for f in 1.0/freqs])
if np.isclose(res, 0.0625):
    host.set_yticks([80, 85, 90, 95, 100, 105])
    host.set_ylim([80, 105])
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    par1.set_ylim([100, 1000])
elif np.isclose(res, 0.125):
    host.set_yticks([30, 35, 40, 45, 50])
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    par1.set_ylim([100, 1000])
elif np.isclose(res, 0.25):
    host.set_yticks([60, 65, 70, 75, 80, 85])
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
elif np.isclose(res, 0.5):
    host.set_yticks([60, 70, 80, 90])
    host.set_ylim([60, 90])
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800, 900])
    par1.set_ylim([100, 900])
elif np.isclose(res, 1.0):
    par1.set_yticks([100, 200, 300, 400, 500, 600, 700, 800])
    par1.set_ylim([200, 800])
for yaxis in (host, par1):
    yticks = yaxis.get_yticks()
    yaxis.set_yticklabels([r"{{{:.0f}}}\%".format(yt) for yt in yticks])
host.set_xlabel("Mesh movement frequency")
host.set_ylabel(r"$\ell_2$ error / uniform mesh")
par1.set_ylabel("CPU time / uniform mesh")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("freq_{:.4f}_{:.1f}".format(res, alpha), "plots", extensions=['pdf'])

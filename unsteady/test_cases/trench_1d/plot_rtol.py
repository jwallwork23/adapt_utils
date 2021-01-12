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
alpha  = 2.0  # TODO


di = os.path.join(os.path.dirname(__file__), 'outputs', 'rtol', '{:.4f}'.format(res))
fnames = ["{:.0e}".format(10**(-i)) for i in range(1, 9)]
resolutions, alphas, tol, time = [], [], [], []
for fname in fnames:
    with open(os.path.join(di, fname), 'r') as f:
        resolutions.append(float(f.readline().split('=')[-1]))
        alphas.append(float(f.readline().split('=')[-1]))
        tol.append(float(f.readline().split('=')[-1]))
        time.append(float(f.readline().split(':')[-1][:-2]))
assert np.allclose(res*np.ones(8), resolutions)
assert np.allclose(alpha*np.ones(8), alphas)
x = list(range(1, len(tol)+1))

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')

# Get discretisation errors
disc_err = []
for rtol in range(1, 9):
    df = pd.read_csv('adapt_output/bed_trench_output_uni_s_{:.4f}_2.0_1.0e-0{:1d}_40.csv'.format(res, rtol))
    disc_err.append(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))
print(disc_err)

# Plot both error and time against rtol
fig, axes = plt.subplots(figsize=(10, 5))
host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes.set_xticks([])
axes.set_yticks([])
axes = plt.gca()
axes.set_xticks(x)
axes.set_xticklabels(["$10^{{-{:d}}}$".format(t) for t in x])
p1, = host.plot(x, disc_err, '--x')
p2, = par1.plot(x, time, '--x')
host.set_xlabel("Relative solver tolerance")
host.set_ylabel(r"Absolute $\ell_2$ error")
par1.set_ylabel(r"Time $[\mathrm s]$")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("rtol_{:.4f}".format(res), "plots", extensions=['pdf', 'png'])

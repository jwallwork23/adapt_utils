import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import numpy as np
import os
import pandas as pd

from adapt_utils.plotting import *


parser = argparse.ArgumentParser()
parser.add_argument("-alpha", help="Monitor function parameter")
parser.add_argument("-freq", help="Mesh movement frequency")
parser.add_argument("-res", help="Resolution in x-direction")
args = parser.parse_args()

alpha = float(args.alpha or 2.0)
freq = int(args.freq or 40)
res = float(args.res or 0.5)

di = os.path.join(os.path.dirname(__file__), 'outputs', 'rtol', '{:.4f}'.format(res))
fnames = ["{:.0e}".format(10**(-i)) for i in range(1, 9)]
resolutions, alphas, tol, time = [], [], [], []
for fname in fnames:
    with open(os.path.join(di, fname), 'r') as f:
        resolutions.append(float(f.readline().split('=')[-1]))
        alphas.append(float(f.readline().split('=')[-1]))
        tol.append(float(f.readline().split('=')[-1]))
        time.append(float(f.readline().split(':')[-1][:-2]))
assert np.allclose(res*np.ones(8), resolutions), resolutions
assert np.allclose(alpha*np.ones(8), alphas), alphas
x = list(range(1, len(tol)+1))

# Turn time into a percentage of the 1.0e-08 result
time = 100*np.array(time)/time[-1]

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')
nrm = np.sqrt(np.sum(df_real['bath']**2))

# Get discretisation errors
disc_err = []
fname = fname = 'adapt_output/bed_trench_output_uni_s_{:.4f}_{:.1f}_1.0e-0{:1d}_{:d}.csv'
for rtol in range(1, 9):
    df = pd.read_csv(fname.format(res, alpha, rtol, freq))
    disc_err.append(100*np.sqrt(np.sum((df['bath'] - df_real['bath'])**2))/nrm)

# Plot both error and time against rtol
fig, axes = plt.subplots(figsize=(8, 5))
host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes.set_xticks([])
axes.set_yticks([])
axes = plt.gca()
axes.set_xticks(x)
axes.set_xticklabels(["$10^{{-{:d}}}$".format(t) for t in x])
p1, = host.plot(x, disc_err, '-o', linewidth=2)
p2, = par1.plot(x, time, '-o', linewidth=2)
host.set_xlabel("Relative solver tolerance")
<<<<<<< HEAD
host.set_ylabel(r"Relative $\ell_2$ error")
=======
host.set_ylabel(r"Relative $\ell^2$ error")
>>>>>>> origin/master
par1.set_ylabel("CPU time / maximum")
if np.isclose(res, 0.0625):
    yticks = [43, 44, 45, 46, 47, 48]
    host.set_yticks(yticks)
    host.set_yticklabels([r"{{{:.0f}}}\%".format(yt) for yt in yticks])
elif np.isclose(res, 0.125):
    host.set_yticklabels([r"{{{:.1f}}}\%".format(yt) for yt in host.get_yticks()])
elif np.isclose(res, 0.25):
    yticks = host.get_yticks()
    host.set_yticks(yticks)
    host.set_yticklabels([r"{{{:.2f}}}\%".format(yt) for yt in yticks])
elif np.isclose(res, 0.5):
    host.set_yticklabels([r"{{{:.5f}}}\%".format(yt) for yt in host.get_yticks()])
yticks = [20, 40, 60, 80, 100]
par1.set_yticks(yticks)
par1.set_yticklabels([r"{{{:.0f}}}\%".format(yt) for yt in yticks])
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("rtol_{:.4f}".format(res), "plots", extensions=['pdf'])

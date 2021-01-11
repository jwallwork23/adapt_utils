import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import pandas as pd

from adapt_utils.plotting import *


# TODO: Combine with gather_rtol

alpha = 2.0
res = 0.5

# Read time data from output logs
tol = []
time = []
for i, line in enumerate(open('outputs/rtol/err_{:.4f}_{:.1f}.csv'.format(res, alpha), 'r')):
    if i == 0:
        continue
    words = line.split(',')
    tol.append(int(words[0][-1]))
    time.append(float(words[3]))
x = list(range(len(tol)))

# Get high resolution data
df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')

# Get discretisation errors
disc_err = []
for rtol in range(1, 9):
    df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.5000_2.0_1.0e-0{:1d}.csv'.format(rtol))
    disc_err.append(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

# Plot both error and time against rtol
fig, axes = plt.subplots(figsize=(10, 5))
host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes.set_xticks([])
axes.set_yticks([])
axes = plt.gca()
axes.set_xticks(x)
axes.set_xticklabels(["$10^{{-{:d}}}$".format(t) for t in tol])
p1, = host.plot(x, disc_err, '--x')
p2, = par1.plot(x, time, '--x')
host.set_xlabel("Relative solver tolerance")
host.set_ylabel(r"Absolute $\ell_2$ error")
par1.set_ylabel(r"Time $[\mathrm s]$")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("rtol", "plots", extensions=['pdf', 'png'])

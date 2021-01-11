import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import math

from adapt_utils.plotting import *


# TODO: Combine with gather_rtol

alpha = 2.0
res = 0.5

tol = []
total_err = []
time = []
for i, line in enumerate(open('outputs/rtol/err_{:.1f}_{:.1f}.csv'.format(res, alpha), 'r')):
    if i == 0:
        continue
    words = line.split(',')
    tol.append(int(words[0][-1]))
    total_err.append(float(words[1]))
    time.append(float(words[3]))
tol = tol[:-1]
x = list(range(len(tol)))
disc_err = [err - total_err[-1] for err in total_err[:-1]]
time = time[:-1]

host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes = plt.gca()
axes.set_xticks(x)
axes.set_xticklabels(["$10^{{-{:d}}}$".format(t) for t in tol])
axes.set_yticks([0, 5.0e-05, 1.0e-04])
axes.set_yticklabels(["0", r"$5\times10^{-5}$", r"$10^{-4}$"])
p1, = host.plot(x, disc_err, '--x')
p2, = par1.plot(x, time, '--x')
host.set_xlabel("Relative solver tolerance")
host.set_ylabel(r"$\ell_2$ error")
par1.set_ylabel(r"Time $[\mathrm s]$")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("rtol", "plots", extensions=[])

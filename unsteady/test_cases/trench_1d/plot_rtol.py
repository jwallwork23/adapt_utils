import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

from adapt_utils.plotting import *


tol = []
err = []
time = []
for i, line in enumerate(open('outputs/rtol/err.csv', 'r')):
    if i == 0:
        continue
    words = line.split(',')
    tol.append(int(words[0][-1]))
    err.append(float(words[1]))
    time.append(float(words[2]))
x = list(range(len(tol)))

host = host_subplot(111, axes_class=AA.Axes)
par1 = host.twinx()
par1.axis["right"].toggle(all=True)
axes = plt.gca()
axes.set_xticks(x)
axes.set_xticklabels(["$10^{{-{:d}}}$".format(t) for t in tol])
p1, = host.plot(x, err, '--x')
p2, = par1.plot(x, time, '--x')
host.set_xlabel("Relative solver tolerance")
host.set_ylabel(r"$\ell_2$ error")
par1.set_ylabel(r"Time $[\mathrm s]$")
host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
plt.draw()
axes.grid(True)
savefig("rtol", "plots", extensions=[])

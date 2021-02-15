import matplotlib.pyplot as plt

from adapt_utils.plotting import *


fig, axes = plt.subplots()
axes.plot([1], [1], 'x', label='Data', color='C1')
axes.plot([1], [1], label='Point evaluation', color='C0')
axes.plot([1], [1], ':', label='Integral', color='C0')
lines, labels = axes.get_legend_handles_labels()
fig2, axes2 = plt.subplots()
legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=3)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
savefig('legend_both', 'plots', bbox_inches=bbox, extensions=['pdf'], tight=False)

fig, axes = plt.subplots()
axes.plot([1], [1], label='Point evaluation', color='C0')
axes.plot([1], [1], label='Integral', color='C1')
lines, labels = axes.get_legend_handles_labels()
fig2, axes2 = plt.subplots()
legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=2)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
savefig('legend_error', 'plots', bbox_inches=bbox, extensions=['pdf'], tight=False)

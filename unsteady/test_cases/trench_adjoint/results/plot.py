#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:43:31 2020

@author: mc4117
"""

import pandas as pd
import pylab as plt

diff_time = pd.read_excel('adapt_output.xlsx', sheet_name = 'diff_time')

error_list = []
time_list = []

for i in diff_time.columns.values[2:-1]:
    error_list.append(diff_time.iloc[1][0])
    time_list.append(diff_time.iloc[9][0])

fig, ax1 = plt.subplots()

plt.gca().ticklabel_format(style='sci', scilimits=(0,1), axis='y')

color = 'tab:red'
ax1.set_xlabel('Number of timesteps per mesh movement')
ax1.set_ylabel('Discretisation error', color=color)
#ax1.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[0][2:-1], ':', label = r'$\alpha = 3$', color=color)
ax1.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[1][2:-1], '-o', label = 'Mesh movement', color=color)
#ax1.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[2][2:-1], ':', label = r'$\alpha = 5$', color=color)
ax1.semilogx(diff_time.columns.values[2:-1], error_list, '--', label = 'Fixed mesh', color = color)
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc = 9, bbox_to_anchor=(0.5, 0.73), title = "Error")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Computational time (s)', color=color)  # we already handled the x-label with ax1
#ax2.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[8][2:-1], ':', label = r'$\alpha = 3$', color=color)
ax2.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[9][2:-1], '-o', label = 'Mesh movement', color=color)
#ax2.semilogx(diff_time.columns.values[2:-1], diff_time.iloc[10][2:-1], ':', label = r'$\alpha = 5$', color=color)
ax2.semilogx(diff_time.columns.values[2:-1], time_list, '--', label = 'Fixed mesh',  color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(loc = 9, bbox_to_anchor=(0.5, 0.95), title = "Cost")
plt.show()

#### Second graph
error_list = []
time_list = []

for i in diff_time.iloc[13][2:-1]:
    error_list.append(diff_time.iloc[14][0])
    time_list.append(diff_time.iloc[22][0])

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of timesteps per mesh movement')
ax1.set_ylabel('Discretisation error', color=color)
#ax1.semilogx(diff_time.iloc[13][2:], diff_time.iloc[14][2:], ':', label = r'$\alpha = 1$', color=color)
ax1.semilogx(diff_time.iloc[13][2:-1], diff_time.iloc[15][2:-1], '-o', label = 'Mesh movement', color=color)
#ax1.semilogx(diff_time.iloc[13][2:], diff_time.iloc[16][2:], ':', label = r'$\alpha = 2$', color=color)
ax1.semilogx(diff_time.iloc[13][2:-1], error_list, '--', label = 'Fixed mesh', color = color)
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(loc = 9, bbox_to_anchor=(0.5, 0.73), title = "Error")
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Computational time (s)', color=color)  # we already handled the x-label with ax1
#ax2.semilogx(diff_time.iloc[13][2:], diff_time.iloc[22][2:], ':', label = r'$\alpha = 1$', color=color)
ax2.semilogx(diff_time.iloc[13][2:-1], diff_time.iloc[23][2:-1], '-o', label = 'Mesh movement', color=color)
#ax2.semilogx(diff_time.iloc[13][2:], diff_time.iloc[24][2:], ':', label = r'$\alpha = 2$', color=color)
ax2.semilogx(diff_time.iloc[13][2:-1], time_list, '--', label = 'Fixed mesh', color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(loc = 9, bbox_to_anchor=(0.5, 0.95), title = "Cost")
plt.show()
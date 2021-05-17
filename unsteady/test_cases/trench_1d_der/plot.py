#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:43:31 2020

@author: mc4117
"""

import pandas as pd
import pylab as plt

import matplotlib
import matplotlib.ticker as mtick

font = {'size'   : 11}

matplotlib.rc('font', **font)


diff_time = pd.read_excel('results.xls', sheet_name = 'mesh movement freq')

error_list = []
time_list = []
fixed_error_list = []
fixed_time_list = []

for i in diff_time.columns.values[2:]:
    error_list.append(diff_time.iloc[0][i])
    time_list.append(diff_time.iloc[2][i])
    fixed_error_list.append(diff_time.iloc[0][diff_time.columns.values[1]])
    fixed_time_list.append(diff_time.iloc[2][diff_time.columns.values[1]])    


fig, ax1 = plt.subplots()

plt.gca().ticklabel_format(style='sci', scilimits=(0,1), axis='y')

color = 'tab:red'
ax1.set_xlabel('Number of timesteps per mesh movement', fontsize = 12)
ax1.set_ylabel('Discretisation error', color=color, fontsize = 12)
ax1.loglog(diff_time.columns.values[2:], error_list/fixed_error_list[0], '-o', label = 'Mesh movement', color=color)
#ax1.loglog(diff_time.columns.values[2:], fixed_error_list/fixed_error_list[0], '--', label = 'Fixed mesh', color = color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yticks([0.11, 0.12, 0.13, 0.14])
ax1.set_yticklabels([0.11, 0.12, 0.13, 0.14])
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.minorticks_off()
plt.legend(loc = 3, bbox_to_anchor=(0.18, 0.65), title = "Error", fontsize = 11)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Computational cost', color=color, fontsize = 12)  # we already handled the x-label with ax1
ax2.loglog(diff_time.columns.values[2:], time_list/fixed_time_list[0], '-o', label = 'Mesh movement', color=color)
#ax2.loglog(diff_time.columns.values[2:], fixed_time_list/fixed_time_list[0], '--', label = 'Fixed mesh',  color = color)
ax2.set_yticks([1, 2, 4, 8])
ax2.set_yticklabels([1, 2, 4, 8])
ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax2.tick_params(axis='y', labelcolor=color)
plt.minorticks_off()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend(loc = 3, bbox_to_anchor=(0.18, 0.8), title = "Cost", fontsize = 11)
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:33:38 2020

@author: mc4117
"""

import pandas as pd
import numpy as np
import pylab as plt

df_real = pd.read_csv('final_result_check_nx3_ny1.csv')

df_test = pd.read_csv('final_result_check_nx2_ny1.csv')

df_test1 = pd.read_csv('final_result_check_nx1.5_ny1.csv')

df_test2 = pd.read_csv('final_result_check_nx1_ny1.csv')

df_test3 = pd.read_csv('final_result_check_nx0.75_ny1.csv')

df_test4 = pd.read_csv('final_result_check_nx0.5_ny0.5.csv')

df_test5 = pd.read_csv('final_result_check_nx0.25_ny0.5.csv')

df_test6 = pd.read_csv('final_result_check_nx0.2_ny0.5.csv')

df_test7 = pd.read_csv('final_result_check_nx0.1_ny0.5.csv')

error_list = []
#error_list.append(0.0)
error_list.append(sum([(df_test['bath'][i] - df_real['bath'][i])**2 for i in range(1, len(df_real))]))
error_list.append(sum([(df_test1['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test2['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test3['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test4['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test5['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test6['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test7['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
print(error_list)

stop
#plt.plot([0.25, 0.5, 2/3, 1, 4/3, 2, 4], error_list)

plt.loglog([0.5, 2/3, 1, 4/3, 2, 4], error_list, '-o')
plt.ylabel('Error norm (m)')
plt.xlabel(r'$\Delta x$ (m)')
plt.show()


logx = np.log([0.5, 2/3, 1, 4/3, 2, 4])
log_error = np.log(error_list)
print(np.polyfit(logx, log_error, 1))

# plot

"""
plt.plot(df_real['x'], df_real['bath'], label = 'nx = 2')
plt.plot(df_test1['x'], df_test1['bath'], label = 'nx = 1')
plt.plot(df_test2['x'], df_test2['bath'], label = 'nx = 0.5')
plt.plot(df_test3['x'], df_test3['bath'], label = 'nx = 0.25')
plt.legend()

"""

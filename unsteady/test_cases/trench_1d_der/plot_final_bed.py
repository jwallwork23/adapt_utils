#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:31:16 2021

@author: mc4117
"""
import pandas as pd
import pylab as plt

coarse_fixed = pd.read_csv(r'bed_trench_output_uni_c_0.1250.csv')
real = pd.read_csv(r'bed_trench_output_uni_c_4.0000.csv')
moving = pd.read_csv('bed_trench_output_uni_s_0.1250_3.0_1.0_1.0.csv')
exp = pd.read_csv('experimental_data.csv', header = None)


plt.plot(real['x'], real['bath'], 'k:', linewidth = 2, label = '"True" value')
plt.plot(moving['x'], moving['bath'], label = 'Mesh movement')
plt.plot(coarse_fixed['x'], coarse_fixed['bath'], '--', label = 'Fixed mesh')
#plt.plot(exp[0], exp[1], label = 'Experimental data')
plt.xlabel('x (m)')
plt.ylabel('Bedlevel (m)')
plt.legend()
plt.show()
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

df_test = pd.read_csv('final_result_nx0.5_4_1_1.csv')

df_test1 = pd.read_csv('final_result_nx0.5_2_0_1.csv')

#df_test2 = pd.read_csv('adapt_output/final_result_nx0.5_2_1_1.csv')

df_test3 = pd.read_csv('final_result_nx0.5_1_1_1.csv')

df_test4 = pd.read_csv('final_result_check_nx0.5_ny0.5.csv')

error_list = []
#error_list.append(0.0)
error_list.append(sum([(df_test['bath'][i] - df_real['bath'][i])**2 for i in range(1, len(df_real))]))
error_list.append(sum([(df_test1['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
#error_list.append(sum([(df_test2['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test3['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))
error_list.append(sum([(df_test4['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))]))

print(error_list)



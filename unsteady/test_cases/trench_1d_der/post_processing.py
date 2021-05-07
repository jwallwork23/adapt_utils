import numpy as np
import pandas as pd

df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.1000.csv')

print('0.1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.1250.csv')

print('0.125')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.2000.csv')

print('0.2')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.2500.csv')

print('0.25')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.4000.csv')

print('0.4')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.5000.csv')

print('0.5')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_0.8000.csv')

print('0.8')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_1.0000.csv')

print('1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('fixed_output/bed_trench_output_uni_c_2.0000.csv')

print('2')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

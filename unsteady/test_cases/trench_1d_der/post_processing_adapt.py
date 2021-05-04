import numpy as np
import pandas as pd

df_real = pd.read_csv('fixed_output/bed_trench_output_uni_c_4.0000.csv')

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_1.0_0.0_1.0.csv')

print('1, 0, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_1.0_0.5_1.0.csv')

print('1, 0.5, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_1.0_1.0_1.0.csv')
print('1, 1, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_3.0_0.0_1.0.csv')

print('3, 0, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_3.0_0.5_1.0.csv')

#print('3, 0.5, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_3.0_1.0_1.0.csv')

#print('3, 1, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_5.0_0.0_1.0.csv')

#print('5, 0, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_5.0_0.5_1.0.csv')

#print('5, 0.5, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_5.0_1.0_1.0.csv')

print('5, 1, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_7.0_1.0_1.0.csv')

#print('7, 1, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_9.0_0.5_1.0.csv')

print('9, 0.5, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

df = pd.read_csv('adapt_output/bed_trench_output_uni_s_0.8000_11.0_1.0_1.0.csv')

print('11, 1, 1')
print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

#df = pd.read_csv('adapt_output/bed_trench_output_uni_s_1.0000_13.0_1.0_1.0.csv')

#print('13, 1, 1')
#print(np.sqrt(sum([(df['bath'][i] - df_real['bath'][i])**2 for i in range(len(df_real))])))

import h5py
import numpy as np


data = {0: {}, 1: {}}

# Read from HDF5
for offset in data.keys():
    infile = h5py.File('outputs/fixed_mesh/hdf5/qoi_offset_{:d}.h5'.format(offset), 'r')
    for key in infile.keys():
        data[offset][key] = np.array(infile[key])
    infile.close()

# Write to LaTeX formatted table
tab = open('data/uniform.tex', 'w')
tab.write(r"\begin{table}[t]" + "\n")
tab.write(r"    \centering" + "\n")
tab.write(r"    \begin{tabular}{|ccc||ccc|}" + "\n")
tab.write(r"        \hline" + "\n")
tab.write(r"        \rowcolor{Gray}" + "\n")
tab.write(r"        \multicolumn{3}{|c||}{Aligned} & \multicolumn{3}{c|}{Offset}\\" + "\n")
tab.write(r"        \rowcolor{Gray}" + "\n")
tab.write(r"        Elements & DoFs & $J_0$ & Elements & DoFs & $J_1$\\" + "\n")
tab.write(r"        \hline" + "\n")
for i in range(len(data[0]['elements'])):
    tex = r"        {:d} & {:d} & ${:.4f}\MW$ & {:d} & {:d} & ${:.4f}\MW$\\" + "\n"
    tab.write(tex.format(data[0]['elements'][i], data[0]['dofs'][i], data[0]['qoi'][i],
                         data[1]['elements'][i], data[1]['dofs'][i], data[1]['qoi'][i]))
tab.write(r"        \hline\noalign{\smallskip}" + "\n")
tab.write(r"    \end{tabular}" + "\n")
tab.write(r"    \caption{Convergence of QoIs $J_0$ and $J_1$ evaluated at finite element solutions on a sequence of meshes generated by uniform refinement of the initial mesh." + "\n")
tab.write(r"    }\label{tab:experiments:setup:convergence}" + "\n")
tab.write(r"\end{table}" + "\n")
tab.close()

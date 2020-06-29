import h5py
import numpy as np


data = {1: {}, 2: {}}

# Read from HDF5
for index in data.keys():
    infile = h5py.File('outputs/fixed_mesh/hdf5/qoi_{:d}.h5'.format(index), 'r')
    for key in infile.keys():
        data[index][key] = np.array(infile[key])
    infile.close()

# Write to LaTeX formatted table
tab = open('data/uniform.tex', 'w')
tab.write(r"\begin{table}[ht!]" + "\n")
tab.write(r"    \centering" + "\n")
tab.write(r"    \begin{tabular}{|m{1.3cm}|m{1.0cm}|m{1.0cm}|m{1.0cm}|m{1.0cm}|}" + "\n")
tab.write(r"        \hline" + "\n")
tab.write(r"        \rowcolor{Gray}" + "\n")
tab.write(r"        Elements & $J_1(\phi)$ & $J_1(\phi_h)$ & $J_2(\phi)$ & $J_2(\phi_h)$\\" + "\n")
tab.write(r"        \hline" + "\n")
for i in range(len(data[1]['elements'])):
    tex = r"        {:d} & {:.4f} & {:.4f} & {:.4f} & {:.4f}\\" + "\n"
    tab.write(tex.format(data[1]['elements'][i], data[1]['qoi_exact'][i], data[1]['qoi'][i],
                         data[2]['qoi_exact'][i], data[2]['qoi'][i]))
tab.write(r"        \hline" + "\n")
tab.write(r"    \end{tabular}" + "\n")
tab.write(r"    \caption{Convergence of QoIs $J_1$ and $J_2$ under analytical and finite element solutions on a sequence of uniform meshes. Columns labelled $J_i(\phi)$ correspond to analytical solutions, whilst columns labelled $J_i(\phi_h)$ correspond to finite element solutions." + "\n")
tab.write(r"    }\label{tab:Telemac}" + "\n")
tab.write(r"\end{table}" + "\n")
tab.close()

import argparse
import h5py
import numpy as np
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('family')
parser.add_argument('-stabilisation')
args = parser.parse_args()

# Get filenames
ext = args.family
assert ext in ('cg', 'dg')
if ext == 'dg':
    if args.stabilisation in ('lf', 'LF', 'lax_friedrichs'):
        ext += '_lf'
else:
    if args.stabilisation in ('su', 'SU'):
        ext += '_su'
    if args.stabilisation in ('supg', 'SUPG'):
        ext += '_supg'
infname = 'qoi_{:s}'.format(ext)
outfname = 'uniform_{:s}.tex'.format(ext)

# Read from HDF5
data = {'aligned': {}, 'offset': {}}
inpath = os.path.join(os.path.dirname(__file__), 'outputs', 'fixed_mesh', 'hdf5')
for key in data:
    with h5py.File(os.path.join(inpath, '{:s}_{:s}.h5'.format(infname, key)), 'r') as infile:
        data[key]['elements'] = np.array(infile['elements'])
        data[key]['qoi'] = np.array(infile['qoi'])
        data[key]['qoi_exact'] = np.array(infile['qoi_exact'])

# Write to LaTeX formatted table
outpath = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(outpath, outfname), 'w') as tab:
    tab.write(r"\begin{table}[ht!]" + "\n")
    tab.write(r"    \centering" + "\n")
    tab.write(r"    \begin{tabular}{|m{1.3cm}|m{1.0cm}|m{1.0cm}|m{1.0cm}|m{1.0cm}|}" + "\n")
    tab.write(r"        \hline" + "\n")
    tab.write(r"        \rowcolor{Gray}" + "\n")
    tab.write(r"        Elements & $J_1(\phi)$ & $J_1(\phi_h)$ & $J_2(\phi)$ & $J_2(\phi_h)$\\" + "\n")
    tab.write(r"        \hline" + "\n")
    for i in range(len(data['aligned']['elements'])):
        tex = r"        {:7d} & {:7.5f} & {:7.5f} & {:7.5f} & {:7.5f}\\" + "\n"
        tab.write(tex.format(data['aligned']['elements'][i], data['aligned']['qoi_exact'][i], data['aligned']['qoi'][i],
                             data['offset']['qoi_exact'][i], data['offset']['qoi'][i]))
    tab.write(r"        \hline" + "\n")
    tab.write(r"    \end{tabular}" + "\n")
    tab.write(r"    \caption{Convergence of QoIs $J_1$ and $J_2$ under analytical and finite element solutions on a sequence of uniform meshes. Columns labelled $J_i(\phi)$ correspond to analytical solutions, whilst columns labelled $J_i(\phi_h)$ correspond to finite element solutions." + "\n")
    tab.write(r"    }\label{tab:Telemac}" + "\n")
    tab.write(r"\end{table}" + "\n")

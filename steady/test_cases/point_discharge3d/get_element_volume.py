import argparse
import pyvista as pv
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('approach')
approach = parser.parse_args().approach

fpath = os.path.join(os.path.dirname(__file__), 'outputs', approach)
mesh = pv.UnstructuredGrid(os.path.join(fpath, "tracer_1.vtu"))
print("Number of cells    = {:d}".format(mesh.n_cells))
print("Number of vertices = {:d}".format(mesh.n_points))
sized = mesh.compute_cell_quality(quality_measure='aspect_ratio')
aspect_ratios = sized.cell_arrays['CellQuality']
print("Min aspect ratio   = {:.4e}".format(aspect_ratios.min()))
print("Max aspect ratio   = {:.4e}".format(aspect_ratios.max()))

sized = mesh.compute_cell_sizes()
cell_volumes = np.abs(sized.cell_arrays["Volume"])
print("Min cell volume    = {:.4e}".format(cell_volumes.min()))
print("Max cell volume    = {:.4e}".format(cell_volumes.max()))

print("Number of aspect ratios over 20 = {:d}".format(len(aspect_ratios[aspect_ratios > 20])))

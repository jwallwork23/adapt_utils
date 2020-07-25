from thetis import *

import pylab as plt
import pandas as pd
import numpy as np
import time
import datetime

from adapt_utils.unsteady.test_cases.beach_sed_model.options import BeachOptions
from adapt_utils.unsteady.solver import AdaptiveProblem

t1 = time.time()

nx = 4.0
ny = 2.0

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

inputdir = 'hydrodynamics_beach_l_sep_nx_' + str(nx*220)

kwargs = {
    'approach': 'fixed_mesh',
    'nx': nx,
    'ny': ny,
    'plot_pvd': True,
    'input_dir': inputdir,
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
    'friction': 'manning'
}

op = BeachOptions(**kwargs)
swp = AdaptiveProblem(op)

t1 = time.time()
swp.solve_forward()
t2 = time.time()

print(t2 - t1)

xaxisthetis1 = []
baththetis1 = []

for i in np.linspace(0, 219, 220):
    xaxisthetis1.append(i)
    baththetis1.append(-swp.solver_obj.fields.bathymetry_2d.at([i, 5]))
df = pd.concat([pd.DataFrame(xaxisthetis1, columns = ['x']), pd.DataFrame(baththetis1, columns = ['bath'])], axis = 1)
df.to_csv("final_result_nx" + str(nx) + "_ny2.csv", index = False)

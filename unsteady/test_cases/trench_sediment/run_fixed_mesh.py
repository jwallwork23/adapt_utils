from adapt_utils.unsteady.test_cases.trench_sediment.options import TrenchSedimentOptions
from adapt_utils.unsteady.solver import AdaptiveProblem
from thetis import *

import pandas as pd
import time
import datetime

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
outputdir = 'outputs' + st

kwargs = {
    'approach': 'fixed_mesh',
    'nx': 4,
    'ny': 2,
    'plot_pvd': True,
    'input_dir': 'hydrodynamics_trench_4',
    'output_dir': outputdir,
    # Spatial discretisation
    'family': 'dg-dg',
    'stabilisation': None,
    'use_automatic_sipg_parameter': True,
}

op = TrenchSedimentOptions(**kwargs)
swp = AdaptiveProblem(op)
swp.solve_forward()

new_mesh = RectangleMesh(16*5*5, 5*1, 16, 1.1)

bath = Function(FunctionSpace(new_mesh, "CG", 1)).project(swp.fwd_solutions_bathymetry[0])

data = pd.read_csv('~/Documents/adapt_utils/test_cases/trench_sed_model/experimental_data.csv', header=None)



datathetis = []
bathymetrythetis1 = []
diff_thetis = []
for i in range(len(data[0].dropna())):
    datathetis.append(data[0].dropna()[i])
    bathymetrythetis1.append(-bath.at([np.round(data[0].dropna()[i], 3), 0.55]))
    diff_thetis.append((data[1].dropna()[i] - bathymetrythetis1[-1])**2)

df = pd.concat([pd.DataFrame(datathetis, columns=['x']), pd.DataFrame(bathymetrythetis1, columns = ['bath'])], axis = 1)

print("L2 norm: ")
print(np.sqrt(sum(diff_thetis)))

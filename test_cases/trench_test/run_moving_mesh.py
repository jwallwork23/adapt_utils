from thetis import *

import pylab as plt
import pandas as pd
import numpy as np

from adapt_utils.test_cases.trench_test.options import TrenchOptions
from adapt_utils.swe.tsunami.solver import TsunamiProblem

op = TrenchOptions(approach='monge_ampere',
                    plot_timeseries=False,
                    plot_pvd=True,
                    debug=False,
                    nonlinear_method='relaxation',
                    # nonlinear_method='quasi_newton',  # FIXME
                    num_adapt=1,
                    qoi_mode='inundation_volume',
                    friction = 'nikuradse',
                    nx=0.5,
                    ny = 1,
                    r_adapt_rtol=1.0e-3)

tp = TsunamiProblem(op, levels=0)
tp.setup_solver()


def gradient_interface_monitor(mesh, alpha=200):
    """
    Monitor function focused around the steep_gradient (budd acta numerica)

    NOTE: Defined on the *computational* mesh.

    :kwarg alpha: controls the size of the dense region surrounding the coast.
    """
    P1 = FunctionSpace(mesh, "CG", 1)
    
    eta = tp.solution.split()[1]
    
    b = tp.solver_obj.fields.bathymetry_2d
    current_mesh = eta.function_space().mesh()
    P1_current = FunctionSpace(current_mesh, "CG", 1)
    bath_dx = interpolate(b.dx(0), P1_current)
    bath_dy = interpolate(b.dx(1), P1_current)
    norm = interpolate(pow(bath_dx, 2) + pow(bath_dy, 2), P1_current)
    norm_proj = project(norm, P1)
    #import ipdb; ipdb.set_trace()
    

    return sqrt(1.0 + alpha*norm_proj)

tp.monitor_function = gradient_interface_monitor
tp.solve(uses_adjoint=False)

xaxisthetis1 = []
bathymetrythetis1 = []

for i in np.linspace(0,15.8, 80):
    xaxisthetis1.append(i)
    bathymetrythetis1.append(-tp.solver_obj.fields.bathymetry_2d.at([i, 0.55]))

df = pd.concat([pd.DataFrame(xaxisthetis1), pd.DataFrame(bathymetrythetis1)], axis = 1)

df.to_csv('bed_trench_adap.csv')

not_adapted_mesh = pd.read_csv('bed_trench_output.csv')
plt.plot(not_adapted_mesh['0'], not_adapted_mesh['0.1'], label = 'not adapted mesh')

data = pd.read_excel('../../../Trench/recreatepaperrun1.xlsx', sheet_name = 'recreatepaperrun', header = None)
diff_15 = pd.read_excel('../../../Trench/extra_diffusion.xlsx')

plt.scatter(data[0], data[1], label = 'Experimental Data')

thetisdf = pd.read_csv('../../../Trench/Sensitivity Analysis/linux_morfacfactor_ten_bed_new_one_diff15.csv')
plt.plot(thetisdf['0'], thetisdf['0.1'], label = 'Thetis')

plt.plot(diff_15['x'][diff_15['y'] == 0.55], -diff_15['diff 0.15 diff factors'][diff_15['y'] == 0.55], label = 'Sisyphe')
plt.plot(xaxisthetis1, bathymetrythetis1, '.', linewidth = 2, label = 'adapted mesh')
plt.legend()
plt.show()

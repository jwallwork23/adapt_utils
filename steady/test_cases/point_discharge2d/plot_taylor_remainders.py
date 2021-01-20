from pyadjoint.verification import convergence_rates

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpltools import annotation
import numpy as np

from adapt_utils.plotting import *


matplotlib.rcParams['font.size'] = 20

h = 0.1
eps = [h*1.0e-02*0.5**i for i in range(4)]
residuals = {'discrete': [], 'continuous': []}
residuals['discrete'].append([
    2.0244178409719186e-07, 5.081280615308816e-08, 1.2728657995378043e-08, 3.1853567321401867e-09
])
residuals['continuous'].append([
    1.095765382175206e-06, 5.982907769831108e-07, 3.118231335727214e-07, 1.5909053905190954e-07
])
residuals['discrete'].append([
    4.508814770912812e-06, 1.1317578844417006e-06, 2.835124206257715e-07, 7.094995570335126e-08
])
residuals['continuous'].append([
    8.293787813682925e-06, 3.0242444058267573e-06, 1.2297556813182999e-06, 5.440715860496154e-07
])
residuals['discrete'].append([
    6.127615165375214e-06, 1.5381121102696197e-06, 3.853090878624184e-07, 9.642522062976481e-08
])
residuals['continuous'].append([
    1.3974125494486587e-05, 5.461367274825306e-06, 2.3469366701402616e-06, 1.0772390117686864e-06
])
residuals['discrete'].append([
    5.808259661399962e-06, 1.4579537923056166e-06, 3.652293078943965e-07, 9.140022984999033e-08
])
residuals['continuous'].append([
    6.9868471897618694e-06, 2.0472475564865702e-06, 6.598761899848733e-07, 2.3872367089522873e-07
])
residuals['discrete'].append([
    5.81198537314006e-06, 1.4588900648100753e-06, 3.6546404359773786e-07, 9.145898605834445e-08
])
residuals['continuous'].append([
    6.668471156736819e-06, 1.887132956608455e-06, 5.795854894969277e-07, 1.9851970900793936e-07
])
residuals['discrete'].append([
    5.805551556901933e-06, 1.4572751798995476e-06, 3.650594523393197e-07, 9.135764368877614e-08
])
residuals['continuous'].append([
    6.004695626335823e-06, 1.5568472146164922e-06, 4.14845469697792e-07, 1.1625065236801229e-07
])

rates = {'discrete': [], 'continuous': []}
dofs = [2121, 8241, 32481, 128961, 513921, 2051841]
for mode in residuals:
    for i in range(6):
        rates[mode].append(convergence_rates(residuals[mode][i], eps, show=False))
    rates[mode + '_min'] = [np.min(rate) for rate in rates[mode]]
    rates[mode + '_max'] = [np.max(rate) for rate in rates[mode]]

fig, axes = plt.subplots()
axes.semilogx(dofs, rates['continuous_min'], '--x', color='C0')
axes.semilogx(dofs, rates['continuous_max'], ':x', color='C0')
axes.set_xlabel("Degrees of freedom")
axes.set_ylabel("Convergence rate")
axes.grid(True)
savefig("taylor_remainder_convergence", "plots", extensions=["pdf"])

fig, axes = plt.subplots(figsize=(9, 4))
axes.loglog(eps, residuals['discrete'][0], '--x', label='Discrete')
axes.loglog(eps, residuals['continuous'][0], '--x', label='Continuous')
annotation.slope_marker((2.2e-04, 4.0e-07), 1, invert=True, ax=axes, size_frac=0.2)
annotation.slope_marker((5.8e-04, 2.0e-08), 2, invert=False, ax=axes, size_frac=0.2)
axes.set_xlabel("Step length")
axes.set_ylabel("Taylor remainder")
axes.grid(True, which='both')
axes.legend(fontsize=20, loc='center left')
savefig("taylor_remainder_0", "plots", extensions=["pdf"])

fig, axes = plt.subplots(figsize=(9, 4))
axes.loglog(eps, residuals['discrete'][4], '--x', label='Discrete')
axes.loglog(eps, residuals['continuous'][4], '--x', label='Continuous')
annotation.slope_marker((5.8e-04, 8.0e-07), 2, invert=False, ax=axes, size_frac=0.2)
axes.set_xlabel("Step length")
axes.set_ylabel("Taylor remainder")
axes.grid(True, which='both')
axes.legend(fontsize=20, loc='upper left')
savefig("taylor_remainder_4", "plots", extensions=["pdf"])

fig, axes = plt.subplots(figsize=(9, 4))
axes.loglog(eps, residuals['discrete'][5], '--x', label='Discrete')
axes.loglog(eps, residuals['continuous'][5], '--x', label='Continuous')
annotation.slope_marker((5.8e-04, 8.0e-07), 2, invert=False, ax=axes, size_frac=0.2)
axes.set_xlabel("Step length")
axes.set_ylabel("Taylor remainder")
axes.grid(True, which='both')
axes.legend(fontsize=20, loc='upper left')
savefig("taylor_remainder_5", "plots", extensions=["pdf"])

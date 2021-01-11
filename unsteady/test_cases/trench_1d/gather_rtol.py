import numpy as np
import os


# TODO: Combine with plot_rtol

res = 0.5

di = os.path.join(os.path.dirname(__file__), 'outputs', 'rtol', '{:.4f}'.format(res))
fnames = ["{:.0e}".format(10**(-i)) for i in range(1, 9)]
res, alpha, rtol, time, total_error, discretisation_error = [], [], [], [], [], []
for fname in fnames:
    with open(os.path.join(di, fname), 'r') as f:
        res.append(float(f.readline().split('=')[-1]))
        alpha.append(float(f.readline().split('=')[-1]))
        rtol.append(float(f.readline().split('=')[-1]))
        time.append(float(f.readline().split(':')[-1][:-2]))
        total_error.append(float(f.readline().split(':')[-1]))
        discretisation_error.append(float(f.readline().split(':')[-1]))
assert np.isclose(res, np.average(res))
alpha = np.average(alpha)
with open(os.path.join(di, 'err_{:.4f}_{:.1f}.csv'.format(res, alpha)), 'w+') as f:
    f.write('tolerance,total error,discretisation error,time\n')
    for tol, terr, derr, t in zip(rtol, total_error, discretisation_error, time):
        f.write('{:.1e},{:.4e},{:.4e},{:1f}\n'.format(tol, terr, derr, t))

from thetis import create_directory

import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
import argparse

plt.style.use('seaborn-talk')

parser = argparse.ArgumentParser()
parser.add_argument("dates")
parser.add_argument("runs")
args = parser.parse_args()

di = os.path.join(os.path.dirname(__file__), 'outputs/qmesh')
dates = args.dates.split(',')
runs = args.runs.split(',')
if len(dates) < len(runs):
    dates = [dates[0] for r in runs]
assert len(dates) == len(runs)

qois_old = None
for date, run in zip(dates, runs):
    filename = os.path.join(di, '-'.join([date, 'run', run]), 'log')
    use = False
    elements, qois = [], []
    nonlinear = False
    with open(filename, 'r') as logfile:
        for line in logfile:
            words = line.split()
            if words[0] == 'Elements' and words[1] == 'QoI':
                use = True
                continue
            if words[0] == 'nonlinear':
                nonlinear = words[-1] == 'True'
            if use:
                elements.append(int(words[0]))
                qois.append(float(words[1]))
    qois = np.array(qois)
    # init_diff = np.zeros(len(qois)) if qois_old is None else qois[0] - qois_old[0]
    # qois_old = qois
    # plt.semilogx(elements, qois - init_diff, label='Nonlinear SWEs' if nonlinear else 'Linear SWEs')
    plt.semilogx(elements, qois, linestyle='--', marker='o', label='Nonlinear SWEs' if nonlinear else 'Linear SWEs')
plt.xlabel("Element count")
plt.ylabel(r"Quantity of interest, $J(\mathbf{u},\eta)$")
plt.legend()

date = datetime.date.today()
date = '{:d}-{:d}-{:d}'.format(date.year, date.month, date.day)
di = os.path.join(di, '{:s}-runs-{:s}'.format(date, '-'.join([r for r in runs])))
if not os.path.exists(di):
    create_directory(di)
plt.savefig(os.path.join(di, 'qoi_convergence.png'))

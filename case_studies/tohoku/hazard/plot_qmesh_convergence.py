import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

from adapt_utils.io import create_directory
from adapt_utils.plotting import *  # NOQA


# --- Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("dates")
parser.add_argument("runs")
args = parser.parse_args()


# --- Set parameters

di = os.path.join(os.path.dirname(__file__), 'outputs', 'qmesh')
dates = args.dates.split(',')
runs = args.runs.split(',')
if len(dates) < len(runs):
    dates = [dates[0] for r in runs]
assert len(dates) == len(runs)
qois_old = None


# --- Load data and plot

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
di = create_directory(os.path.join(di, '{:s}-runs-{:s}'.format(date, '-'.join([r for r in runs]))))
plt.savefig(os.path.join(di, 'qoi_convergence.png'))

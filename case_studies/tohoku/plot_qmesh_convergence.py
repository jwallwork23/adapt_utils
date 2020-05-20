import matplotlib.pyplot as plt
import os
import argparse

plt.style.use('seaborn-talk')

parser = argparse.ArgumentParser()
parser.add_argument("dates")
parser.add_argument("runs")
args = parser.parse_args()

di = os.path.join(os.path.dirname(__file__), 'outputs/qmesh')
dates = args.dates.split(',')
runs = args.runs.split(',')
assert len(dates) == len(runs)

for date, run in zip(dates, runs):
    filename = os.path.join(di, '-run'.join([date, run]), 'log')
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
    plt.semilogx(elements, qois, label='Nonlinear SWEs' if nonlinear else 'Linear SWEs')
plt.xlabel("Element count")
plt.ylabel(r"Quantity of interest, $J(\mathbf{u},\eta)$")
plt.legend()
plt.show()

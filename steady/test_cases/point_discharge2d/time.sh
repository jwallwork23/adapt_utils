#!/bin/bash

level=$1

t="0.0000,0.0000"
for i in {1..10}
do
	python3 run_continuous_adjoint.py -level $level -time 1 > tmp
	t=$(python3 -c """
import numpy as np
times = np.array([float(t) for t in '$t'.split(',')])
times += np.array([float(t) for t in open('tmp', 'r').readlines()])
print(times[0], ',', times[1])
        """)
done
python3 -c """
import numpy as np
times = np.array([float(t) for t in '$t'.split(',')])/10
print('Continuous adjoint timings averaged over 10 runs')
print('Forward: {:.4f}s, Adjoint: {:.4f}s'.format(*times))
"""

t="0.0000,0.0000"
for i in {1..10}
do
	python3 run_adjoint.py -level $level -time 1 > tmp
	t=$(python3 -c """
import numpy as np
times = np.array([float(t) for t in '$t'.split(',')])
times += np.array([float(t) for t in open('tmp', 'r').readlines()])
print(times[0], ',', times[1])
        """)
done
python3 -c """
import numpy as np
times = np.array([float(t) for t in '$t'.split(',')])/10
print('\nDiscrete adjoint timings averaged over 10 runs')
print('Forward: {:.4f}s, Adjoint: {:.4f}s'.format(*times))
"""

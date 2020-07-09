import os
import numpy as np
import scipy.interpolate


__all__ = ["sample_timeseries"]


def sample_timeseries(gauge, sample=1):
    """
    Interpolate from gauge data. Since the data is provided at regular intervals, we use linear
    interpolation between the data points.

    Since some of the timeseries are rather noisy, there is an optional `sample` parameter, which
    averages over the specified number of datapoints before interpolating.
    """
    time_prev = 0.0
    fpath = os.path.dirname(__file__)
    num_lines = sum(1 for line in open(os.path.join(fpath, gauge+'.dat'), 'r'))
    t, d, running = [], [], []
    with open(os.path.join(fpath, gauge+'.dat'), 'r') as f:
        for i in range(num_lines):
            time, dat = f.readline().split()
            time, dat = float(time), float(dat)
            if np.isnan(dat):
                continue
            running.append(dat)
            if i % sample == 0 and i > 0:
                t.append(0.5*(time + time_prev))
                d.append(np.mean(running))
                running = 0
                time_prev = time
                running = []

    interp = scipy.interpolate.interp1d(t, d, bounds_error=False, fill_value='extrapolate')
    init = interp(0.0)

    def shifted(tau):
        return interp(tau) - init

    return shifted

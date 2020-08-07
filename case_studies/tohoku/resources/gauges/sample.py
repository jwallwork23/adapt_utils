import os
import numpy as np


__all__ = ["get_raw_data", "sample_timeseries"]


# TODO: REPLACE WITH VERSION IN OPTIONS CLASS

def extract_data(gauge):
    """
    Extract time and elevation data from file as NumPy arrays.

    Note that this isn't *raw* data because it has been converted to appropriate units using
    `preproc.py`.
    """
    fpath = os.path.dirname(__file__)
    data_file = os.path.join(fpath, gauge + '.dat')
    if not os.path.exists(data_file):
        raise IOError("Requested timeseries for gauge '{:s}' cannot be found.".format(gauge))
    times, data = [], []
    with open(data_file, 'r') as f:
        for line in f:
            time, dat = line.split()
            times.append(float(time))
            data.append(float(dat))
    return np.array(times), np.array(data)


def sample_timeseries(gauge, sample=1):
    """
    Interpolate from gauge data. Since the data is provided at regular intervals, we use linear
    interpolation between the data points.

    Since some of the timeseries are rather noisy, there is an optional `sample` parameter, which
    averages over the specified number of datapoints before interpolating.
    """
    from scipy.interpolate import interp1d

    times, data = extract_data(gauge)
    time_prev = 0.0
    sampled_times, sampled_data, running = [], [], []
    for time, dat in zip(times, data):
        if np.isnan(dat):
            continue
        running.append(dat)
        if i % sample == 0 and i > 0:
            sampled_times.append(0.5*(time + time_prev))
            sampled_data.append(np.mean(running))
            time_prev = time
            running = []

    interp = interp1d(sampled_times, sampled_data, bounds_error=False, fill_value='extrapolate')
    init = interp(0.0)

    def shifted(tau):
        return interp(tau) - init

    return shifted

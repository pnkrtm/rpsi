import numpy as np
from scipy.interpolate import CubicSpline


def reflection_time(t0, x, vnmo):
    t = np.sqrt(t0**2 + x**2/vnmo**2)
    return t


def sample_trace(trace, time, dt):
    before = int(np.floor(time/dt))
    N = trace.size
    samples = np.arange(before - 1, before + 3)
    if any(samples < 0) or any(samples >= N):
        amplitude = None
    else:
        times = dt*samples
        amps = trace[samples]
        interpolator = CubicSpline(times, amps)
        amplitude = interpolator(time)
    return amplitude


def nmo_correction(cmp, dt, offsets, velocities):
    nmo = np.zeros_like(cmp)
    nsamples = cmp.shape[0]
    times = np.arange(0, nsamples*dt, dt)
    for i, t0 in enumerate(times):
        for j, x in enumerate(offsets):
            t = reflection_time(t0, x, velocities[i])
            amplitude = sample_trace(cmp[:, j], t, dt)
            if amplitude is not None:
                nmo[i, j] = amplitude
    return nmo
import numpy as np

from Objects.Seismic.Seismogram import Seismogram


def rmse_per_column(matr_obs: np.ndarray, matr_mod: np.ndarray, trace_weights: np.ndarray=None):
    if trace_weights is None:
        # trace_weights = np.ones(matr_obs.shape[0])
        trace_weights = np.arange(0, matr_obs.shape[0])
        # trace_weights = trace_weights**2
        trace_weights = trace_weights / trace_weights.sum()

    def calculate_increments(values):
        arr_1 = values[1::]
        arr_2 = values[:-1]

        return arr_1 - arr_2

    def calcing_per_trace(observed, modeled):
        use_increments = False

        ind1 = np.nonzero(observed)[0]
        ind2 = np.nonzero(modeled)[0]

        ind = list(set(ind1) | set(ind2))

        obs = observed
        mod = modeled

        # obs = observed[ind]
        # mod = modeled[ind]

        obs *= 1
        mod *= 1

        diff_func = lambda x, y: np.sqrt(np.mean((x - y)**2))
        # diff_func = lambda x, y: np.mean((x - y) ** 2)
        # diff_func = lambda x, y: np.mean(abs((x - y) / x))

        diff = diff_func(obs, mod)

        if use_increments:
            incr_obs = calculate_increments(obs)
            incr_mod = calculate_increments(mod)

            diff = 0.5 * diff + 0.5 * diff_func(incr_obs, incr_mod)

        return diff

    diffs = [calcing_per_trace(mo, mm) for mo, mm in zip(matr_obs, matr_mod)]

    return np.average(diffs, weights=trace_weights)


def get_matrices_diff(seism_observed: Seismogram, seism_modeled: Seismogram,
                      indexes_start: np.ndarray, indexes_stop: np.ndarray, weights: np.ndarray=None):
    vals_obs = seism_observed.get_values_matrix()
    vals_mod = seism_modeled.get_values_matrix()

    vals_obs = np.array([vo[indexes_start[j]: indexes_stop[j]] for j, vo in enumerate(vals_obs)])
    vals_mod = np.array([vm[indexes_start[j]: indexes_stop[j]] for j, vm in enumerate(vals_mod)])

    return rmse_per_column(vals_obs, vals_mod, weights)
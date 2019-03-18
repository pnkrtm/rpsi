import numpy as np
import multiprocessing as mp
import gc

from ForwardModeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.Optimizators.Optimizations import DifferentialEvolution, DifferentialEvolution_parallel, DualAnnealing
from Objects.Seismogram import Seismogram
from Inversion.Utils.Tools import OptimizeHelper
from Exceptions.exceptions import ErrorAchievedException


def rmse_per_column(matr_obs: np.ndarray, matr_mod: np.ndarray, trace_weights: np.ndarray=None):
    if trace_weights is None:
        trace_weights = np.ones(matr_obs.shape[0])
        trace_weights = trace_weights / trace_weights.sum()

    def calcing_per_trace(observed, modeled):
        ind1 = np.nonzero(observed)[0]
        ind2 = np.nonzero(modeled)[0]

        ind = list(set(ind1) | set(ind2))

        # diff = np.sqrt(np.mean((observed[ind] - modeled[ind])**2)) / np.mean(observed[ind])
        # diff = np.sqrt(np.mean((observed[ind] - modeled[ind]) ** 2))
        # diff = np.mean(abs(observed[ind] - modeled[ind])) / np.mean(observed[ind])
        diff = np.sqrt(np.mean((observed - modeled) ** 2))

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


def func_to_optimize_mp_helper(args):

    return func_to_optimize(**args)


def func_to_optimize(model_opt, seismogram_observed, params_all, params_to_optimize, params_bounds,
                     start_indexes, stop_indexes,
                     helper, parallel, pool, trace_weights=None, normalize=False):
    if parallel:
        # определние величины популяции
        njobs = int(len(model_opt) / len(model_opt[0]))

        input_args = []

        for i in range(njobs):
            model_opt_ = model_opt[i, :]

            input_args.append(
                {
                    'seismogram_observed': seismogram_observed,
                    'model_opt': model_opt_,
                    'params_all': params_all,
                    'params_to_optimize': params_to_optimize,
                    'params_bounds': params_bounds,
                    'start_indexes': start_indexes,
                    'stop_indexes': stop_indexes,
                    'helper': helper,
                    'parallel': False,
                    'pool': None,
                    'trace_weights': trace_weights,
                    'normalize': normalize
                }
            )

        result_errors = pool.map(func_to_optimize_mp_helper, input_args)

        gc.collect()

        return result_errors

    model_opt_2 = model_opt

    params_all_ = {}

    for key in list(params_all.keys()):
        if type(params_all[key]) == type([]):
            params_all_[key] = params_all[key].copy()

        else:
            params_all_[key] = params_all[key]

    for m, p, b in zip(model_opt_2, params_to_optimize, params_bounds):
        if normalize:
            val = b[0] + (b[1] - b[0]) * m

        else:
            val = m

        params_all_[list(p.keys())[0]][list(p.values())[0]] = val

    observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**params_all_)

    error = get_matrices_diff(seismogram_observed, seismogram_p, start_indexes, stop_indexes, trace_weights)

    if np.isnan(error):
        error = 99999

    print(error)

    if helper is not None:
        helper.add_error(error)
        if helper.need_to_stop():
            raise ErrorAchievedException(model_opt)

    return error


def inverse(optimizers, error, params_all, params_to_optimize, params_bounds,
            seismogram_observed_p, seismogram_observed_s,
            start_indexes, stop_indexes,
            trace_weights=None, normalize=True, logpath=None):

    if type(optimizers[0]) in [DifferentialEvolution, DifferentialEvolution_parallel, DualAnnealing]:
        data_start = params_bounds
        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error, logpath=logpath)
        helper.in_use = False

        if type(optimizers[0]) == DifferentialEvolution_parallel:
            parallel = optimizers[0].parallel

            if parallel:
                ncpu = mp.cpu_count()
                nproc = int(ncpu * 2)

                pool = mp.Pool(nproc)

            else:
                parallel = False
                pool = None
        else:
            parallel = False
            pool = None

        args = [seismogram_observed_p, params_all, params_to_optimize,
                params_bounds, start_indexes, stop_indexes,
                     helper, parallel, pool, trace_weights, normalize]

        if normalize:
            data_start_de = [[0, 1]] * len(data_start)
            param_bounds_lbfgs = [[0, 1]] * len(data_start)

        else:
            data_start_de = data_start
            param_bounds_lbfgs = params_bounds

        try:
            result_model = optimizers[0].optimize(func_to_optimize, data_start_de, args)
            start_model = result_model

        except ErrorAchievedException as e:
            start_model = e.model

        finally:
            if pool:
                pool.close()
                pool.join()

        print('======  LBFGS optimization started! =======')

        # Disable stopping by error
        helper.in_use = False
        # Disable parallel
        args[-4] = False

        helper.log_message('======  LBFGS optimization started! =======')

        print(start_model)

        result_model = optimizers[1].optimize(func_to_optimize, start_model, param_bounds_lbfgs, tuple(args))

    if normalize:
        result_model = [b[0] + (b[1] - b[0]) * rm for rm, b in zip(result_model, params_bounds)]

    return result_model

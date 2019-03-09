import numpy as np
import multiprocessing as mp
import gc

from ForwardModeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.Optimizators.Optimizations import DifferentialEvolution, DifferentialEvolution_parallel
from Objects.Seismogram import Seismogram
from Inversion.Utils.Tools import OptimizeHelper
from Exceptions.exceptions import ErrorAchievedException


def rmse_per_column(matr_obs: np.ndarray, matr_mod: np.ndarray, trace_weights: np.ndarray=None):
    if trace_weights is None:
        trace_weights = np.ones(matr_obs.shape[0])
        trace_weights = trace_weights / trace_weights.sum()

    def sqrt_mean(a):
        return np.sqrt(np.mean(a))

    def substract_nonzero_vals(a, b):
        """
        Функция для подсчета разницы только между ненулевыми значениями трассы
        :param a:
        :param b:
        :return:
        """
        ind1 = np.nonzero(a)[0]
        ind2 = np.nonzero(b)[0]

        ind = list(set(ind1) | set(ind2))

        return a[ind] - b[ind]

    # matr_diff = (matr_obs - matr_mod) ** 2
    matr_diff = [substract_nonzero_vals(mo, mm)**2 for mo, mm in zip(matr_obs, matr_mod)]
    # TODO необходимо чекнуть, праивльно ли задана ось, вдоль которой производить применение функции
    # matr_diff = np.apply_along_axis(sqrt_mean, 1, matr_diff)
    matr_diff = [sqrt_mean(md) for md in matr_diff]

    return np.average(matr_diff, weights=trace_weights)


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
                     helper, parallel, pool, trace_weights=None):
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
                    'trace_weights': trace_weights
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

    for m, p in zip(model_opt_2, params_to_optimize):
        params_all_[list(p.keys())[0]][list(p.values())[0]] = m

    observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**params_all_)

    error = get_matrices_diff(seismogram_observed, seismogram_p, start_indexes, stop_indexes, trace_weights)

    if np.isnan(error):
        error = 99999

    helper.add_error(error)

    print(error)

    if helper.need_to_stop():
        raise ErrorAchievedException(model_opt)

    return error


def inverse(optimizers, error, params_all, params_to_optimize, params_bounds,
            seismogram_observed_p, seismogram_observed_s,
            start_indexes, stop_indexes,
            trace_weights=None):

    if type(optimizers[0]) in [DifferentialEvolution, DifferentialEvolution_parallel]:
        data_start = params_bounds
        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error)

        if type(optimizers[0]) == DifferentialEvolution_parallel:
            parallel = optimizers[0].parallel

            if parallel:

                helper.in_use = False

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
                     helper, parallel, pool, trace_weights]

        try:
            result_model = optimizers[0].optimize(func_to_optimize, data_start, args)
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
        args[-3] = False

        result_model = optimizers[1].optimize(func_to_optimize, start_model, params_bounds, tuple(args))

    return result_model

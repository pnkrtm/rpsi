import numpy as np
import multiprocessing as mp
import gc
import datetime

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
                     helper, trace_weights=None, normalize=False, show_tol=True):
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

    if show_tol:
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
    """
    Функция единичной инверсии одной точки
    :param optimizers: массив применяемых оптимизаторов
    :param error: значение ошибки, при которой инверсия останавливатеся
    :param params_all: словарь аргументов для решения ПЗ
    :param params_to_optimize: словарь параметров, которые необходимо подбирать
    :param params_bounds: границы подбора параметров
    :param seismogram_observed_p: наблюденные данные p-волн
    :param seismogram_observed_s: наблюденные данные s-волн
    :param start_indexes: массив индексов, ограничивающих сверху область сейсмограммы для расчета невязки
    :param stop_indexes: массив индексов, ограничивающих снизу область сейсмограммы для расчета невязки
    :param trace_weights: массив весов для трасс
    :param normalize: флаг, указывающий нужно ли нормировать подбираемые параметры
    :param logpath: путь до файла, в который пишутся логи
    :return:
    """
    show_tol = True

    data_start = params_bounds
    helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error, logpath=logpath)
    helper.in_use = True

    for i in range(len(optimizers)):
        optimizers[i].helper = helper

    helper.log_message(f"{str(datetime.datetime.now())} Optimization starts!")

    args = tuple([seismogram_observed_p, params_all, params_to_optimize,
            params_bounds, start_indexes, stop_indexes,
                 helper, trace_weights, normalize, show_tol])

    if normalize:
        data_start_opt = [[0, 1]] * len(data_start)
        param_bounds_opt = [[0, 1]] * len(data_start)

    else:
        data_start_opt = data_start
        param_bounds_opt = params_bounds

    optimizator_start_params = {
        "func": func_to_optimize,
        "x0": data_start_opt,
        "bounds": param_bounds_opt,
        "args": args
    }

    for opt in optimizers:
        try:
            result_model = opt.optimize(**optimizator_start_params)
            optimizator_start_params["x0"] = result_model

        except ErrorAchievedException as e:
            result_model = e.model
            helper.log_message(f'{str(datetime.datetime.now())} Random error achieved!! =)))')
            break

    if normalize:
        result_model = [b[0] + (b[1] - b[0]) * rm for rm, b in zip(result_model, params_bounds)]

    helper.log_message(f'{str(datetime.datetime.now())} Optimization finished!')

    return result_model

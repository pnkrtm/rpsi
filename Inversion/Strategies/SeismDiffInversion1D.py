import datetime

import numpy as np

from Exceptions.bad_calcs import BadCalcBaseException
from Exceptions.exceptions import ErrorAchievedException
from ForwardModeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.Utils.Tools import OptimizeHelper
from Inversion.Utils.Calcs import get_matrices_diff


def get_params_dict_copy(params_all):
    params_all_ = {}

    for key in list(params_all.keys()):
        if isinstance(params_all[key], (list, np.ndarray)):
            params_all_[key] = params_all[key].copy()

        else:
            params_all_[key] = params_all[key]

    return params_all_


def change_values_in_params_dict(params_all, params_vals, params_keys, params_bounds, normalize):
    for m, p, b in zip(params_vals, params_keys, params_bounds):
        if normalize:
            val = b[0] + (b[1] - b[0]) * m

        else:
            val = m

        params_all[list(p.keys())[0]][list(p.values())[0]] = val

    return params_all


def func_to_optimize_mp_helper(args):

    return func_to_optimize(**args)


def func_to_optimize(model_opt, seismogram_observed, params_all, params_to_optimize, params_bounds,
                     start_indexes, stop_indexes,
                     helper, trace_weights=None, normalize=False, show_tol=True):
    try:
        params_all_ = get_params_dict_copy(params_all)

        params_all_ = change_values_in_params_dict(params_all_, model_opt, params_to_optimize, params_bounds, normalize)

        observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**params_all_)

        error = get_matrices_diff(seismogram_observed, seismogram_p, start_indexes, stop_indexes, trace_weights)

        # Добавляем минимизацию к-тов оражения
        aip_1 = model.get_param('aip', index_finish=-1)
        aip_2 = model.get_param('aip', index_start=1)
        rp = (aip_2 - aip_1) / (aip_2 + aip_1)

        error = 1.0 * error + 0.0 * np.sum(abs(rp))

        if np.isnan(error):
            error = 99999

    except OverflowError as e:
        error = 99999

    except BadCalcBaseException as e:
        error = 99999

    # except Warning:
    #     error = 99999

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
            helper.log_message(f"{str(datetime.datetime.now())} Optimizer finished!")
            optimizator_start_params["x0"] = result_model

        except ErrorAchievedException as e:
            result_model = e.model
            helper.log_message(f'{str(datetime.datetime.now())} Random error achieved!! =)))')
            break

    if normalize:
        result_model = [b[0] + (b[1] - b[0]) * rm for rm, b in zip(result_model, params_bounds)]

    helper.log_message(f'{str(datetime.datetime.now())} Optimization finished!')

    return result_model


def inverse_per_layer(optimizers, error, params_all, params_to_optimize, params_bounds,
            seismogram_observed_p, seismogram_observed_s,
            start_indexes, stop_indexes,
            trace_weights=None, normalize=True, logpath=None):
    # добавить больше гипербол для ограничения сейсмограмм

    layers_to_inverse = list(set([list(po.values())[0] for po in params_to_optimize]))
    params_all_local = get_params_dict_copy(params_all)

    # Подбираем отдельно для первого и второго слоев
    params_to_optimize_local = [po for po in params_to_optimize if list(po.values())[0] == layers_to_inverse[0] or
                                                                        list(po.values())[0] == layers_to_inverse[1]]
    res_model = inverse(optimizers, error, params_all_local, params_to_optimize_local, params_bounds,
                        seismogram_observed_p, seismogram_observed_s, start_indexes, stop_indexes,
                        trace_weights, normalize, logpath)

    params_all_local = change_values_in_params_dict(params_all_local, res_model, params_to_optimize,
                                                    params_bounds, normalize)

    result = []

    # Подбираем для всех остальных слоев (3+)
    for layer in layers_to_inverse[2:]:
        params_to_optimize_local = [po for po in params_to_optimize if list(po.values())[0] == layer]

        res_model = inverse(optimizers, error, params_all_local, params_to_optimize_local, params_bounds,
                            seismogram_observed_p, seismogram_observed_s, start_indexes, stop_indexes,
                            trace_weights, normalize, logpath)

        result = np.append(result, res_model)

        params_all_local = change_values_in_params_dict(params_all_local, res_model, params_to_optimize,
                                                        params_bounds, normalize)

    return result


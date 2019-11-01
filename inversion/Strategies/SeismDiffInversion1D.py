import datetime

import numpy as np

from Exceptions.bad_calcs import BadCalcBaseException
from Exceptions.exceptions import ErrorAchievedException
from fmodeling.ForwardProcessing1D import forward_with_trace_calcing
from inversion.Utils.Tools import OptimizeHelper
from inversion.Utils.Calcs import get_matrices_diff


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


def func_to_optimize(model_opt, placeholders, forward_params, helper=None, show_tol=True):
    try:
        forward_params['model'].set_optimization_option(model_opt)

        observe, seismic = forward_with_trace_calcing(**forward_params)

        errors = []
        for key in seismic.keys():
            ph = placeholders[key]

            errors.append(get_matrices_diff(ph.seismogram, seismic[key]["seismogram"], ph.start_indexes,
                                            ph.stop_indexes, ph.trace_weights))
        error = np.mean(errors)
        # Добавляем минимизацию к-тов оражения
        aip_1 = forward_params['model'].get_single_param('aip', index_finish=-1)
        aip_2 = forward_params['model'].get_single_param('aip', index_start=1)
        rp = (aip_2 - aip_1) / (aip_2 + aip_1)

        error = 1 * error + 0 * np.sum(abs(rp))

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


def inverse(optimizers, error, placeholders, forward_params, logpath=None, scale=None):
    """
    Функция единичной инверсии одной точки

    :return:
    """
    show_tol = True

    forward_params['model'].scale = scale

    data_start = forward_params['model'].get_optimization_option('val', vectorize=True)

    min_bound = forward_params['model'].get_optimization_option('min', vectorize=True)
    max_bound = forward_params['model'].get_optimization_option('max', vectorize=True)

    param_bounds = np.column_stack((min_bound.T, max_bound.T))

    if error is not None:
        if isinstance(error, float):
            error = [error] * len(optimizers)
        elif isinstance(error, (list, np.ndarray, tuple)):
            if len(error) != len(optimizers):
                raise ValueError("Bad errors list length!")
        else:
            raise TypeError("Unknown error type!")

        helper = OptimizeHelper(nerrors=len(data_start), error_to_stop=error, logpath=logpath)
        helper.in_use = True

    else:
        helper = OptimizeHelper(nerrors=len(data_start), logpath=logpath)
        helper.in_use = False

    for i in range(len(optimizers)):
        optimizers[i].helper = helper

    helper.log_message(f"{str(datetime.datetime.now())} Optimization starts!")

    args = (placeholders, forward_params, helper, show_tol)

    optimizator_start_params = {
        "func": func_to_optimize,
        "x0": data_start,
        "bounds": param_bounds,
        "args": args
    }

    for opt in optimizers:
        try:
            result_model = opt.optimize(**optimizator_start_params)
            helper.log_message(f"{str(datetime.datetime.now())} Optimizer finished!")
            optimizator_start_params["x0"] = result_model

        except ErrorAchievedException as e:
            result_model = e.model
            optimizator_start_params["x0"] = result_model
            print(f'{str(datetime.datetime.now())} Random error achieved!! =)))')
            helper.log_message(f'{str(datetime.datetime.now())} Random error achieved!! =)))')


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


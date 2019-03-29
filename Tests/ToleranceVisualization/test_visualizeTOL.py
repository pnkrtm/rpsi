import numpy as np
from matplotlib import pyplot as plt

from Inversion.DataIO import read_input_fp_file, read_input_ip_file
from Inversion.Strategies.SeismDiffInversion1D import func_to_optimize


def single_param_visualize():
    model_folder = 'model_example_4'
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize, \
        observation_params = read_input_fp_file(model_folder)

    dx = float(observation_params['dx'])
    nrec = observation_params['nrec']

    x_rec = [i * dx for i in range(1, nrec + 1)]
    dt = observation_params['dt']

    forward_base_params = {
        'x_rec': x_rec,
        'display_stat': False,
        'visualize_seismograms': False,
        'use_p_waves': observation_params['use_p_waves'],
        'use_s_waves': observation_params['use_s_waves'],
        'dt': dt,
        'trace_len': observation_params['trace_len'],
    }
    forward_input_params = dict(forward_base_params)
    forward_input_params.update(params_all_dict)

    seismogram_observed, err, optimizers, start_indexes, stop_indexes = read_input_ip_file(model_folder,
                                                                                           np.array(x_rec),
                                                                                           observation_params['dt'])

    npoints = 100

    points_vals = np.arange(0, 1, 0.01)
    tols = [func_to_optimize([pv], seismogram_observed, forward_input_params, params_to_optimize, bounds_to_optimize,
                             start_indexes, stop_indexes, None, False, None, normalize=True) for pv in points_vals]

    x = bounds_to_optimize[0][0] + points_vals * (bounds_to_optimize[0][1] - bounds_to_optimize[0][0])
    # x = points_vals
    y = tols

    plt.plot(x, y)
    plt.show()


def double_param_visualize():
    model_folder = 'model_example_4'
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize, \
        observation_params = read_input_fp_file(model_folder)

    dx = float(observation_params['dx'])
    nrec = observation_params['nrec']

    x_rec = [i * dx for i in range(1, nrec + 1)]
    dt = observation_params['dt']

    forward_base_params = {
        'x_rec': x_rec,
        'display_stat': False,
        'visualize_seismograms': False,
        'use_p_waves': observation_params['use_p_waves'],
        'use_s_waves': observation_params['use_s_waves'],
        'dt': dt,
        'trace_len': observation_params['trace_len'],
    }
    forward_input_params = dict(forward_base_params)
    forward_input_params.update(params_all_dict)

    seismogram_observed, err, optimizers, start_indexes, stop_indexes = read_input_ip_file(model_folder,
                                                                                           np.array(x_rec),
                                                                                           observation_params['dt'])

    # def func_2D(param_key_1, param_val_1, bounds_1, param_key_2, param_val_2, bounds_2, params_all):
    #     params_all[list(param_key_1.keys())[0]][list(param_key_1.values())[0]] = \
    #         bounds_1[0] + (bounds_1[1] - bounds_1[0]) * param_val_1
    #     params_all[list(param_key_2.keys())[0]][list(param_key_2.values())[0]] = \
    #         bounds_2[0] + (bounds_2[1] - bounds_2[0]) * param_val_2
    #
    #     return params_all

    npoints = 30

    x = np.arange(0, 1, 1 / npoints)
    y = np.arange(0, 1, 1 / npoints)

    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()]).T

    tols = [func_to_optimize([p[0], p[1]], seismogram_observed, forward_input_params, params_to_optimize, bounds_to_optimize,
                             start_indexes, stop_indexes, None, False, None, normalize=True) for p in positions]

    tols = np.reshape(tols, (len(X), len(Y))).T

    X = bounds_to_optimize[0][0] + X * (bounds_to_optimize[0][1] - bounds_to_optimize[0][0])
    Y = bounds_to_optimize[1][0] + Y * (bounds_to_optimize[1][1] - bounds_to_optimize[1][0])

    plt.contourf(X, Y, tols, levels=20)
    plt.colorbar()

    plt.show()

def main():
    double_param_visualize()


if __name__ == '__main__':
    main()
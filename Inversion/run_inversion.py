import argparse
import sys
import time
import os
import numpy as np

sys.path.append('../')

from Inversion.DataIO import read_input_file, write_output_file, read_input_fp_file, read_input_ip_file
from Visualization.Drawing import draw_seismogram
from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from Inversion.Utils.visualize_inversion_results import write_averaged_result, plot_histogram_by_all_results
from Inversion.Optimizators.Optimizations import LBFGSBOptimization, DifferentialEvolution_parallel
from Inversion.Strategies.Inversion1D import inverse_universal, inverse_universal_shots
from Inversion.Strategies.SeismDiffInversion1D import inverse

def main(input_folder, dx, nx, use_rays_p, use_rays_s,
        use_reflection_p, use_reflection_s, forward_type, noise):

    file_name = os.path.join(input_folder, 'input_fp.json')
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(file_name)

    x_rec = [i * dx for i in range(1, nx)]

    optimizers = [
        # DifferentialEvolution(popsize=6, maxiter=10, atol=1000, init='random', polish=False),
        DifferentialEvolution_parallel(popsize=20, maxiter=30, init='random', strategy='best1bin', polish=False,
                                       tol=0.000001, mutation=0.5, recombination=0.9, parallel=True),
        # DifferentialEvolution_parallel(polish=False, init='random', strategy='best1bin'),
        LBFGSBOptimization()
    ]

    error = 0.1

    if forward_type == 0:

        forward_input_params = {
            'x_rec': x_rec,
            'display_stat': False,
            'visualize_res': False,
            'calc_reflection_p': use_rays_p,
            'calc_reflection_s': use_rays_s,
            'noise': noise
        }

        layer_weights = [
            0.7,
            0.7,
            1,
            0.95,
            0.8,
            0.7,
            0.7
        ]

        forward_input_params.update(params_all_dict)

        observe, model, rays_observed_p, rays_observed_s, reflection_observed_p, reflection_observed_s = \
            forward(**forward_input_params)

        forward_input_params['noise'] = False

        inversion_start_time = time.time()

        inversed_model = inverse_universal(optimizers, error, forward_input_params, params_to_optimize,
                                           bounds_to_optimize,
                                           rays_observed_p, rays_observed_s,
                                           reflection_observed_p, reflection_observed_s,
                                           opt_type='de',
                                           use_rays_p=use_rays_p, use_rays_s=use_rays_s,
                                           use_reflection_p=use_reflection_p, use_reflection_s=use_reflection_s,
                                           layer_weights=layer_weights
                                           )



    elif forward_type == 1:

        forward_input_params = {
            'x_rec': x_rec,
            'display_stat': False,
            'visualize_res': False,
            'use_p_waves': use_reflection_p,
            'use_s_waves': use_reflection_s,
            'dt': 3e-03,
            'trace_len': 1500,
        }

        forward_input_params.update(params_all_dict)

        seismogram_observed_p, seismogram_observed_s = forward_with_trace_calcing(**forward_input_params)

        inversion_start_time = time.time()

        inversed_model = inverse(optimizers, error, forward_input_params, params_to_optimize, bounds_to_optimize,
                                 seismogram_observed_p, seismogram_observed_s, start_indexes, stop_indexes,
                                 opt_type='de', use_p_waves=True, use_s_waves=False, trace_weights=None)


    inversion_end_time = time.time()
    inversion_duration = (inversion_end_time - inversion_start_time) / 60

    print('Inversion duration: {} min'.format(inversion_duration))

    write_output_file(input_folder, params_all_dict, inversed_model, params_to_optimize, inversion_duration)


def main_2(model_folder, draw_pics):
    inversion_start_time = time.time()

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

    seismogram_observed, err, optimizers, start_indexes, stop_indexes = read_input_ip_file(model_folder, np.array(x_rec),
                                                                                  observation_params['dt'])

    inversed_model = inverse(optimizers, err, forward_input_params, params_to_optimize, bounds_to_optimize,
                seismogram_observed, None, start_indexes, stop_indexes)

    inversion_end_time = time.time()
    inversion_duration = (inversion_end_time - inversion_start_time) / 60

    result_number = write_output_file(model_folder, params_all_dict, inversed_model, params_to_optimize, inversion_duration)

    if draw_pics:
        # changing start model to result model
        params_all_ = {}
        for key in list(params_all_dict.keys()):
            if type(params_all_dict[key]) == type([]):
                params_all_[key] = params_all_dict[key].copy()

            else:
                params_all_[key] = params_all_dict[key]

        for m, p in zip(inversed_model, params_to_optimize):
            params_all_[list(p.keys())[0]][list(p.values())[0]] = m

        forward_input_params = dict(forward_base_params)
        forward_input_params.update(params_all_)

        observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**forward_input_params)

        picks_folder = os.path.join(model_folder, 'output', f'result_{result_number}')

        draw_seismogram(seismogram_observed, 'p-waves observed', os.path.join(picks_folder, 'p-observed.png'),
                        additional_lines=[
                            [x_rec, start_indexes*dt, {}],
                            [x_rec, stop_indexes * dt, {}],
                        ])

        draw_seismogram(seismogram_p, 'p-waves inverted', os.path.join(picks_folder, 'p-inverted.png'))
        draw_seismogram(seismogram_observed - seismogram_p, 'p-waves difference', os.path.join(picks_folder, 'p-difference.png'))

        write_averaged_result(model_folder)
        plot_histogram_by_all_results(model_folder)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=None, nargs='?',
                        help="Path to source folder")
    parser.add_argument("-dp", "--draw_pics", default=False, nargs='?',
                        help="Flag for pics drawing")

    args = parser.parse_args()

    main_2(args.input_folder, args.draw_pics)

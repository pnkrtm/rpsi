import argparse
import os
import sys
import time

import numpy as np

sys.path.append('../')

from inversion.DataIO import write_output_file, read_input_fp_file, read_input_ip_file, create_res_folder, write_segy
from visualization.Drawing import draw_seismogram, draw_dos_seismograms
from visualization.Models import visualize_model
from fmodelling.forward_proc_1D import forward_with_trace_calcing
from inversion.Utils.visualize_inversion_results import write_averaged_result, plot_histogram_by_all_results
from inversion.Strategies.SeismDiffInversion1D import inverse


def main(model_folder, draw_pics):
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

    result_number = create_res_folder(model_folder)
    result_folder = os.path.join(model_folder, 'output', f'result_{result_number}')

    logpath = os.path.join(result_folder, 'opt.log')

    inversed_model = inverse(optimizers, err, forward_input_params, params_to_optimize, bounds_to_optimize,
                seismogram_observed, None, start_indexes, stop_indexes, logpath=logpath)

    # inversed_model = inverse_per_layer(optimizers, err, forward_input_params, params_to_optimize, bounds_to_optimize,
    #                          seismogram_observed, None, start_indexes, stop_indexes, logpath=logpath)

    inversion_end_time = time.time()
    inversion_duration = (inversion_end_time - inversion_start_time) / 60

    write_output_file(model_folder, params_all_dict, inversed_model, params_to_optimize, inversion_duration, result_number)

    if draw_pics:
        # calculate true model's params
        observe_true, model_true, rays_p_true, rays_s_true, seismogram_p_true, seismogram_s_true = \
            forward_with_trace_calcing(**forward_input_params)

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

        # calculate forward from inverse model
        observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**forward_input_params)

        picks_folder = os.path.join(model_folder, 'output', f'result_{result_number}')

        # draw wellogs
        visualize_model(model_true, model, picks_folder)

        draw_seismogram(seismogram_observed, 'p-waves observed', os.path.join(picks_folder, 'p-observed.png'),
                        additional_lines=[
                            [x_rec, start_indexes*dt, {}],
                            [x_rec, stop_indexes * dt, {}],
                        ])

        draw_seismogram(seismogram_p, 'p-waves inverted', os.path.join(picks_folder, 'p-inverted.png'))

        draw_dos_seismograms(seismogram_observed, seismogram_p, 'p-waves compare',
                             os.path.join(picks_folder, 'p-compare.png'), normalize=False)

        draw_seismogram(seismogram_observed, 'p-waves observed', os.path.join(picks_folder, 'p-observed-wig.png'),
                        wiggles=True, normalize=False)
        draw_seismogram(seismogram_p, 'p-waves inverted', os.path.join(picks_folder, 'p-inverted-wig.png'),
                        wiggles=True, normalize=False)

        write_segy(seismogram_p, os.path.join(result_folder, 'pwaves_inv.sgy'))
        draw_seismogram(seismogram_observed - seismogram_p, 'p-waves difference',
                        os.path.join(picks_folder, 'p-difference.png'), colorbar=True)

        write_segy(seismogram_observed - seismogram_p, os.path.join(result_folder, 'pwaves_diff.sgy'))

        write_averaged_result(model_folder)
        plot_histogram_by_all_results(model_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=None, nargs='?',
                        help="Path to source folder")
    parser.add_argument("-dp", "--draw_pics", default=0, nargs='?',
                        help="Flag for pics drawing")

    args = parser.parse_args()

    main(args.input_folder, bool(args.draw_pics))

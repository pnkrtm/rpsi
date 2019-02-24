import os
import argparse

from ForwardModeling.ForwardProcessing1D import forward_with_trace_calcing
from Inversion.DataIO import read_input_file_ver2, write_segy


def main(input_folder):
    input_filename = os.path.join(input_folder, 'input', 'input.json')

    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize, \
        observation_params, inversion_params = read_input_file_ver2(input_filename)

    dx = float(observation_params['dx'])
    nrec = observation_params['nrec']

    x_rec = [i * dx for i in range(1, nrec + 1)]

    forward_input_params = {
        'x_rec': x_rec,
        'display_stat': True,
        'visualize_res': False,
        'use_p_waves': observation_params['use_p_waves'],
        'use_s_waves': observation_params['use_s_waves'],
        'dt': observation_params['dt'],
        'trace_len': observation_params['trace_len'],
    }

    forward_input_params.update(params_all_dict)

    observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(**forward_input_params)

    write_segy(seismogram_p, os.path.join(input_folder, 'input', 'pwaves.sgy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=None, nargs='?',
                        help="Path to source folder")

    args = parser.parse_args()

    main(args.input_folder)

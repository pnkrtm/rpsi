import os
import argparse

from fmodeling.forward_proc_1D import forward_with_trace_calcing
from inversion.DataIO import read_input_fp_file, write_segy
from Visualization.Drawing import draw_seismogram


def main(model_folder, draw_pics):
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize, \
        observation_params = read_input_fp_file(model_folder)

    dx = float(observation_params['dx'])
    nrec = observation_params['nrec']

    x_rec = [i * dx for i in range(1, nrec + 1)]

    forward_input_params = {
        'x_rec': x_rec,
        'display_stat': True,
        'visualize_seismograms': False,
        'use_p_waves': observation_params['use_p_waves'],
        'use_s_waves': observation_params['use_s_waves'],
        'dt': observation_params['dt'],
        'trace_len': observation_params['trace_len'],
    }

    forward_input_params.update(params_all_dict)

    observe, model, res_seismic = forward_with_trace_calcing(**forward_input_params)

    write_segy(seismogram_p, os.path.join(model_folder, 'input', 'pwaves.sgy'))

    if draw_pics:
        picks_folder = os.path.join(model_folder, 'pics')
        draw_seismogram(seismogram_p, 'p-waves', os.path.join(picks_folder, 'p-observed.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=None, nargs='?',
                        help="Path to source folder")
    parser.add_argument("-dp", "--draw_pics", default=False, nargs='?',
                        help="Flag for pics drawing")

    args = parser.parse_args()

    main(args.input_folder, args.draw_pics)

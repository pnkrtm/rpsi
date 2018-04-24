import argparse
import sys
import time
sys.path.append('../')

from Inversion.DataIO import read_input_file, write_output_file
import numpy as np
from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from Inversion.Optimizations import LBFGSBOptimization, DifferentialEvolution
from Inversion.Inversion1D import inverse_universal, inverse_universal_shots


def main(input_folder, dx, nx, use_rays_p, use_rays_s,
        use_reflection_p, use_reflection_s, forward_type):

    file_name = input_folder + '\\input.json'
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(file_name)

    x_rec = [i * dx for i in range(1, nx)]

    optimizers = [
        DifferentialEvolution(popsize=6, maxiter=10, atol=1000, init='random', polish=False),
        LBFGSBOptimization()
    ]

    error = 0.1

    if forward_type == 0:

        forward_input_params = {
            'x_rec': x_rec,
            'display_stat': False,
            'visualize_res': False,
            'calc_reflection_p': use_rays_p,
            'calc_reflection_s': use_rays_s
        }

        forward_input_params.update(params_all_dict)

        observe, model, rays_observed_p, rays_observed_s, reflection_observed_p, reflection_observed_s = \
            forward(**forward_input_params)

        inversion_start_time = time.time()

        inversed_model = inverse_universal(optimizers, error, forward_input_params, params_to_optimize,
                                           bounds_to_optimize,
                                           rays_observed_p, rays_observed_s,
                                           reflection_observed_p, reflection_observed_s,
                                           opt_type='de',
                                           use_rays_p=use_rays_p, use_rays_s=use_rays_s,
                                           use_reflection_p=use_reflection_p, use_reflection_s=use_reflection_s
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

        inversed_model = inverse_universal_shots(optimizers, error, forward_input_params, params_to_optimize,
                                                 bounds_to_optimize,
                                                 seismogram_observed_p, seismogram_observed_s,
                                                 opt_type='de', use_p_waves=use_rays_p, use_s_waves=use_rays_s
                                                 )


    inversion_end_time = time.time()
    inversion_duration = (inversion_end_time - inversion_start_time) / 60

    print('Inversion duration: {} min'.format(inversion_duration))

    write_output_file(input_folder, params_all_dict, inversed_model, params_to_optimize, inversion_duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", default=None, nargs='?',
                        help="Path to source folder")
    parser.add_argument("-nx", "--nx", default=None, nargs='?',
                        help="Number of x-receivers")
    parser.add_argument("-dx", "--dx", default=None, nargs='?',
                        help="Distance X between receivers")
    parser.add_argument("-rays_p", "--use_rays_p", default=True, nargs='?',
                        help="Use p-rays")
    parser.add_argument("-rays_s", "--use_rays_s", default=True, nargs='?',
                        help="Use s-rays")
    parser.add_argument("-refl_p", "--use_reflection_p", default=True, nargs='?',
                        help="Use p-rays")
    parser.add_argument("-refl_s", "--use_reflection_s", default=True, nargs='?',
                        help="Use s-rays")
    parser.add_argument("-f_t", "--forward_type", default=0, nargs='?',
                        help="Forward type")

    args = parser.parse_args()

    input_folder = args.input_folder
    nx = int(args.nx)
    dx = float(args.dx)

    use_rays_p = args.use_rays_p
    use_rays_s = args.use_rays_s
    use_reflection_p = args.use_reflection_p
    use_reflection_s = args.use_reflection_s
    forward_type = int(args.forward_type)

    main(input_folder, dx, nx, use_rays_p, use_rays_s, use_reflection_p, use_reflection_s, forward_type)

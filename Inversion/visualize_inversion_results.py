from Inversion.DataIO import read_input_file, write_output_file, read_inversion_result_file
from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from matplotlib import pyplot as plt
from Visualization.Seismic import visualize_model1D, visualize_model_wellogs, visualize_rays_model_1D, visualize_time_curves, \
    visualize_reflection_amplitudes, visualize_seismogram

def main(input_folder, dx, nx, use_rays_p, use_rays_s, result_number=1):
    input_file_name = input_folder + '/input.json'
    result_file_name = input_folder + '/result_{}'.format(result_number)
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(input_file_name)
    params_optimized, values_optimized = read_inversion_result_file(result_file_name)

    # without seismograms forwarding
    x_rec = [i * dx for i in range(1, nx)]

    forward_input_params = {
        'x_rec': x_rec,
        'display_stat': False,
        'visualize_res': False,
        'calc_reflection_p': use_rays_p,
        'calc_reflection_s': use_rays_s
    }

    forward_input_params.update(params_all_dict)

    params_all_ = {}

    for key in list(forward_input_params.keys()):
        if type(forward_input_params[key]) == type([]):
            params_all_[key] = forward_input_params[key].copy()

        else:
            params_all_[key] = forward_input_params[key]

    for m, p in zip(values_optimized, params_to_optimize):
        params_all_[list(p.keys())[0]][list(p.values())[0]] = m

    observe_1, model_1, rays_p_1, rays_s_1, reflection_p_1, reflection_s_1 = forward(**forward_input_params)
    observe_2, model_2, rays_p_2, rays_s_2, reflection_p_2, reflection_s_2 = forward(**params_all_)

    max_depth = model_1.get_max_boundary_depth() * 1.2
    dz = 100

    # visualize_model_wellogs(plt, model_2, 'vp', legend_label='vp, m/s')
    # visualize_model_wellogs(plt, model_2, 'vs', legend_label='vs, m/s')
    # visualize_model_wellogs(plt, model_2, 'rho', legend_label='dens, kg/m3')
    #
    # plt.gca().invert_yaxis()
    # # plt.title('Vp welllogs')
    # plt.legend()
    # plt.show()

    # visualize_model1D(plt, model_1, observe_1, max_depth, dz, 'vp', only_boundaries=True)
    # visualize_rays_model_1D(plt, rays_p_1)
    # plt.gca().invert_yaxis()
    # # plt.legend()
    # plt.show()

    # visualize_reflection_amplitudes(plt, reflection_p_1, 'angle')
    # # plt.legend()
    # plt.show()

    visualize_time_curves(plt, model_1, rays_p_1, observe_1)
    plt.show()


    # # with seismogram forwarding
    # forward_input_params_shots = {
    #     'x_rec': x_rec,
    #     'display_stat': False,
    #     'visualize_res': False,
    #     'use_p_waves': True,
    #     'use_s_waves': True,
    #     'dt': 3e-03,
    #     'trace_len': 800,
    # }
    #
    # forward_input_params_shots.update(params_all_dict)
    #
    #
    # seismogram_p_1, seismogram_s_1 = forward_with_trace_calcing(**forward_input_params_shots)
    #
    # visualize_seismogram(plt, seismogram_p_1, normalize=True, wigles=False)
    # plt.show()


if __name__ == '__main__':
    input_folder = '../work_models/'
    dx = 20
    nx = 80
    use_rays_p = True
    use_rays_s = True
    result_number = 10
    model_number = 4
    main(input_folder + 'model_{}/'.format(model_number), dx, nx, use_rays_p, use_rays_s, result_number)
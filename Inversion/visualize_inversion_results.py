from Inversion.DataIO import read_input_file, write_output_file, read_inversion_result_file, get_results_files_list
from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from matplotlib import pyplot as plt
from matplotlib import mlab, rc
from Visualization.Seismic import visualize_model1D, visualize_model_wellogs, visualize_rays_model_1D, visualize_time_curves, \
    visualize_reflection_amplitudes, visualize_seismogram

import os
import numpy as np

font = {
    # 'family': 'normal',
    #     'weight': 'bold',
          'size': 16}

rc('font', **font)

params_names_dict = {
        'rho_m': 'плотность скелета',
        'rho_s': 'плотность глин',
        'rho_f': 'плотность флюида',
        'Kf': 'K флюида',
        'phi': 'пористость',
        'phi_s': 'объем глин'
    }

x_titles_dict = {
    'rho_m': 'кг/м3',
    'rho_s': 'кг/м3',
    'rho_f': 'кг/м3',
    'Kf': 'ГПа',
    'phi': '',
    'phi_s': ''
}

scale_factors_dict = {
    'rho_m': 1000,
    'rho_s': 1000,
    'rho_f': 1000,
    'Kf': 1,
    'phi': 1,
    'phi_s': 1
}


def get_all_results(input_folder):
    params_optimized_all = []
    values_optimized_all = []

    result_files = get_results_files_list(input_folder)

    for file in result_files:
        params_optimized_, values_optimized_ = read_inversion_result_file(
            os.path.join(input_folder, 'result_{}'.format(file)))
        params_optimized_all.append(list(params_optimized_))
        values_optimized_all.append(list(values_optimized_))

    params_optimized_all = np.array(params_optimized_all)
    values_optimized_all = np.array(values_optimized_all)

    nparams = len(params_optimized_all[0])

    return params_optimized_all, values_optimized_all, nparams


def plot_histogram_by_all_results(input_folder):
    input_file_name = input_folder + '/input.json'
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(input_file_name)

    params_optimized_all, values_optimized_all, nparams = get_all_results(input_folder)

    for i in range(nparams):
        param_name = list(params_optimized_all[0][i].keys())[0]
        param_index = int(list(params_optimized_all[0][i].values())[0])

        # if 'phi' in param_name:
        #     hist_width = 5
        # else:
        #     hist_width = 10
        hist_width = 10

        plt.hist(values_optimized_all[:, i]*scale_factors_dict[param_name], hist_width, facecolor='green', alpha=0.75, edgecolor='k')

        true_value = params_all_dict[param_name][param_index]
        plt.xlabel(x_titles_dict[param_name])
        plt.xlim(np.array(bounds_to_optimize[i])*scale_factors_dict[param_name])
        plt.axvline(true_value*scale_factors_dict[param_name], color='r', alpha=0.7, lw=2)

        plt.title('Слой №{} {}'.format(param_index+1, params_names_dict[param_name]), fontsize=18)
        plt.tight_layout()

        # plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(input_folder, 'pictures/{}_{}.png'.format(param_name, param_index)))
        plt.close()


def write_averaged_result(input_folder):
    input_file_name = input_folder + '/input.json'
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(input_file_name)

    params_optimized_, values_optimized_mean = get_averaged_model(input_folder)

    write_output_file(input_folder, params_all_dict, values_optimized_mean, params_to_optimize, file_name='result_average')


def get_averaged_model(input_folder):
    params_optimized_all = []
    values_optimized_all = []
    values_optimized_mean = []
    result_files = get_results_files_list(input_folder)

    for file in result_files:
        params_optimized_, values_optimized_ = read_inversion_result_file(
            os.path.join(input_folder, 'result_{}'.format(file)))
        params_optimized_all.append(list(params_optimized_))
        values_optimized_all.append(list(values_optimized_))

    nparams = len(params_optimized_all[0])
    values_optimized_all = np.array(values_optimized_all)

    for i in range(nparams):
        vals = values_optimized_all[:, i]
        values_optimized_mean.append(np.mean(vals))

    return params_optimized_, values_optimized_mean




def main(input_folder, dx, nx, use_rays_p, use_rays_s, result_number=1):
    input_file_name = input_folder + '/input.json'
    result_file_name = input_folder + '/result_{}'.format(result_number)
    nlayers, params_all_dict, params_to_optimize, bounds_to_optimize = read_input_file(input_file_name)
    params_optimized, values_optimized = get_averaged_model(input_folder)
    # params_optimized, values_optimized = read_inversion_result_file(result_file_name)

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

    visualize_model_wellogs(plt, model_1, 'vp', legend_label='истиная модель')
    visualize_model_wellogs(plt, model_2, 'vp', legend_label='результат подбора', linestyle='--')
    plt.gca().invert_yaxis()
    plt.title('vp', fontsize=18)
    plt.ylabel('глубина, м')
    plt.xlabel('скорость, м/с')
    plt.tight_layout()
    plt.savefig(os.path.join(input_folder, 'pictures/vp_average.png'))
    plt.close()

    visualize_model_wellogs(plt, model_1, 'vs', legend_label='истиная модель')
    visualize_model_wellogs(plt, model_2, 'vs', legend_label='результат подбора', linestyle='--')
    plt.gca().invert_yaxis()
    plt.title('vs', fontsize=18)
    plt.ylabel('глубина, м')
    plt.xlabel('скорость, м/с')
    plt.tight_layout()
    plt.savefig(os.path.join(input_folder, 'pictures/vs_average.png'))
    plt.close()

    visualize_model_wellogs(plt, model_1, 'rho', legend_label='истиная модель')
    visualize_model_wellogs(plt, model_2, 'rho', legend_label='результат подбора', linestyle='--')
    plt.gca().invert_yaxis()
    plt.title('rho', fontsize=18)
    plt.ylabel('глубина, м')
    plt.xlabel('плотность, кг/м3')
    plt.tight_layout()
    plt.savefig(os.path.join(input_folder, 'pictures/rho_average.png'))
    plt.close()

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

    for i in range(nlayers-1):
        visualize_reflection_amplitudes(plt, reflection_p_1, reflection_index=i, absc='angle')
        visualize_reflection_amplitudes(plt, reflection_p_2, reflection_index=i, absc='angle', linestyle='--')
        plt.title('отражение от подошвы {} слоя для p-волн'.format(i+1), fontsize=18)
        plt.ylabel('к-т отражения')
        plt.xlabel('угол падения, рад')
        plt.tight_layout()
        plt.savefig(os.path.join(input_folder, 'pictures/amplitudes_vp_{}.png'.format(i)))
        plt.close()

        visualize_time_curves(plt, model_1, rays_p_1, observe_1, depth_index=i)
        visualize_time_curves(plt, model_2, rays_p_2, observe_2, depth_index=i, linestyle='--')
        plt.title('отражение от подошвы {} слоя для p-волн'.format(i + 1), fontsize=18)
        plt.ylabel('время, сек')
        plt.xlabel('удаление, м')
        plt.tight_layout()
        plt.savefig(os.path.join(input_folder, 'pictures/times_vp_{}.png'.format(i)))
        plt.close()

        visualize_reflection_amplitudes(plt, reflection_s_1, reflection_index=i, absc='angle')
        visualize_reflection_amplitudes(plt, reflection_s_2, reflection_index=i, absc='angle', linestyle='--')
        plt.title('отражение от подошвы {} слоя для s-волн'.format(i + 1), fontsize=18)
        plt.ylabel('амплитуда')
        plt.xlabel('угол падения, рад')
        plt.tight_layout()
        plt.savefig(os.path.join(input_folder, 'pictures/amplitudes_vs_{}.png'.format(i)))
        plt.close()

        visualize_time_curves(plt, model_1, rays_s_1, observe_1, depth_index=i)
        visualize_time_curves(plt, model_2, rays_s_2, observe_2, depth_index=i, linestyle='--')
        plt.title('отражение от подошвы {} слоя для s-волн'.format(i + 1), fontsize=18)
        plt.ylabel('время, сек')
        plt.xlabel('удаление, м')
        plt.tight_layout()
        plt.savefig(os.path.join(input_folder, 'pictures/times_vs_{}.png'.format(i)))
        plt.close()
    # # plt.legend()
    # plt.show()

    # visualize_time_curves(plt, model_1, rays_p_1, observe_1)
    # plt.show()


    # with seismogram forwarding
    forward_input_params_shots = {
        'x_rec': x_rec,
        'display_stat': False,
        'visualize_res': False,
        'use_p_waves': True,
        'use_s_waves': True,
        'dt': 3e-03,
        'trace_len': 800,
    }

    forward_input_params_shots.update(params_all_dict)


    seismogram_p_1, seismogram_s_1 = forward_with_trace_calcing(**forward_input_params_shots)

    # visualize_seismogram(plt, seismogram_p_1, normalize=False, wigles=False, gain=1)
    # plt.show()


if __name__ == '__main__':
    input_folder = '../work_models/'
    dx = 20
    nx = 80
    use_rays_p = True
    use_rays_s = True
    result_number = 10
    model_number = 4
    plot_histogram_by_all_results(input_folder + 'model_{}/'.format(model_number))
    main(input_folder + 'model_{}/'.format(model_number), dx, nx, use_rays_p, use_rays_s, result_number)
    write_averaged_result(input_folder + 'model_{}/'.format(model_number))
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp

from ForwardModeling.RockPhysics.Models import model_calculation, simple_model_1
from Objects.Models.Models import SeismicModel1D
from Objects.Observation import Observation, Source, Receiver
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays, calculate_rays_for_layer
from ForwardModeling.Seismic.Dynamic.Reflection import calculate_reflections, calculate_reflection_for_depth
from Visualization.Seismic import visualize_model1D, visualize_rays_model_1D, visualize_time_curves, visualize_reflection_amplitudes


def forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
            display_stat=False, visualize_res=True,
            calc_rays_p=True, calc_rays_s=True,
            calc_reflection_p=True, calc_reflection_s=True):

    '''

    :param nlayers: Кол-во слоев одномерной модели
    :param Km: Массив массивов модулей сжатия матрикса (для каждого слоя задается массив из составляющих его минералов)
    :param Gm: Массив массивов модулей сдвига матрикса (для каждого слоя задается массив из составляющих его минералов)
    :param Ks: Массив модулей сжатия глины
    :param Gs: Массив модулей сдвига глины
    :param Kf: Массив модулей сжатия флюида
    :param phi: Массив пористости
    :param phi_s: Массив К глинистости (кол-во глины)
    :param rho_s: Массив плотностей глин
    :param rho_f: Массив плотностнй флюида
    :param rho_m: Массив массивов плотностей минералов матрикса
    :param h: Массив мощностей
    :param x_rec: Массив приемников
    :return:
    '''

    if display_stat:
        disp_func = lambda x: print(x)

    else:
        disp_func = lambda x: x  # пустая функция при отстутствии написания параметров

    disp_func('Calculating rockphysics model...')
    rp_start_time = time.time()

    vp, vs, rho = model_calculation(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m)

    disp_func('Rockphysics model calculated!')

    model = SeismicModel1D(vp, vs, rho, h)
    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x) for x in x_rec]
    observe = Observation(sources, receivers)

    rays_p = None
    rays_s = None

    rays_start_time = time.time()

    if calc_rays_p:
        disp_func('Calculating p-rays...')
        rays_p = calculate_rays(observe, model, 'vp')

    if calc_rays_s:
        disp_func('Calculating s-rays...')
        rays_s = calculate_rays(observe, model, 'vs')

    disp_func('Rays calculated!')

    reflection_start_time = time.time()

    reflections_p = None
    if calc_reflection_p:
        disp_func('Calculating p-reflections...')
        reflections_p = calculate_reflections(model, rays_p, 'PdPu')

    reflections_s = None
    if calc_reflection_s:
        disp_func('Calculating s-reflections...')
        reflections_s = calculate_reflections(model, rays_s, 'SdSu')

        disp_func('Reflections calculated!')

    calc_stop_time = time.time()

    # print('rp time = {}'.format(rays_start_time - rp_start_time))
    # print('ray tracing time = {}'.format(reflection_start_time - rays_start_time))
    # print('reflection time = {}'.format(calc_stop_time - reflection_start_time))
    # print('all_time = {}'.format(calc_stop_time - rp_start_time))

    if visualize_res:
        max_depth = model.get_max_boundary_depth() * 1.2
        dz = 100
        disp_func('Drawing results...')
        fig, axes = plt.subplots(nrows=3, ncols=2)

        visualize_model1D(axes[2, 0], model, max_depth, dz, 'vp')
        visualize_rays_model_1D(axes[2, 0], rays_p)
        axes[2, 0].set_title('model and rays for p-waves')

        visualize_model1D(axes[2, 1], model, max_depth, dz, 'vs')
        visualize_rays_model_1D(axes[2, 1], rays_s)
        axes[2, 1].set_title('model and rays for s-waves')

        visualize_time_curves(axes[1, 0], model, rays_p, observe)
        axes[1, 0].set_title('time curves for p-waves')
        visualize_time_curves(axes[1, 1], model, rays_s, observe)
        axes[1, 1].set_title('time curves for s-waves')

        if calc_reflection_p:
            visualize_reflection_amplitudes(axes[0, 0], reflections_p, 'angle')
            axes[0, 0].set_title('avo for p-waves')

        if calc_reflection_s:
            visualize_reflection_amplitudes(axes[0, 1], reflections_s, 'angle')
            axes[0, 1].set_title('avo for for s-waves')

        plt.show()

    return observe, model, rays_p, rays_s, reflections_p, reflections_s


# def forward_with_parallel_mp_helper(args)
#
#
# def forward_with_parallel(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, display_stat=False,
#             visualize_res=True, calc_reflection=True):
#     '''
#
#         :param nlayers: Кол-во слоев одномерной модели
#         :param Km: Массив массивов модулей сжатия матрикса (для каждого слоя задается массив из составляющих его минералов)
#         :param Gm: Массив массивов модулей сдвига матрикса (для каждого слоя задается массив из составляющих его минералов)
#         :param Ks: Массив модулей сжатия глины
#         :param Gs: Массив модулей сдвига глины
#         :param Kf: Массив модулей сжатия флюида
#         :param phi: Массив пористости
#         :param phi_s: Массив К глинистости (кол-во глины)
#         :param rho_s: Массив плотностей глин
#         :param rho_f: Массив плотностнй флюида
#         :param rho_m: Массив массивов плотностей минералов матрикса
#         :param h: Массив мощностей
#         :param x_rec: Массив приемников
#         :return:
#         '''
#
#     sources = [Source(0, 0, 0)]
#     receivers = [Receiver(x) for x in x_rec]
#     observe = Observation(sources, receivers)
#
#     input_args = np.column_stack([])
#
#     for i in range(nlayers):
#         vp, vs, rho = model_calculation(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m)
#
#
#         model = SeismicModel1D(vp, vs, rho, h)
#         sources = [Source(0, 0, 0)]
#         receivers = [Receiver(x) for x in x_rec]
#         observe = Observation(sources, receivers)
#
#         rays_p = calculate_rays(observe, model, 'vp')
#
#         rays_s = calculate_rays(observe, model, 'vs')
#
#         if calc_reflection:
#             reflections_p = calculate_reflections(model, rays_p, 'PdPu')
#
#             reflections_s = calculate_reflections(model, rays_s, 'SdSu')



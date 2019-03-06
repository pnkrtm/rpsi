import numpy as np
from scipy.signal import ricker
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import random as rnd

from ForwardModeling.RockPhysics.Models import model_calculation, simple_model_1

from Objects.Models.Models import SeismicModel1D
from Objects.Observation import Observation, Source, Receiver
from Objects.Seismogram import Trace, Seismogram
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays, calculate_rays_for_layer
from ForwardModeling.Seismic.Dynamic.Reflection import calculate_reflections
from ForwardModeling.Seismic.Dynamic.Refraction import calculate_refractions
from Visualization.Seismic import visualize_model1D, visualize_model_wellogs, visualize_rays_model_1D, \
    visualize_time_curves, \
    visualize_reflection_amplitudes, visualize_seismogram


def add_noise_rays(rays, depths):
    for d in depths[1:]:
        rays_ = [r for r in rays if r.reflection_z == d]
        times_ = np.array([r.time for r in rays_])
        mean_time = np.mean(times_)

        for r in rays_:
            percent_coeff = 0.1
            # погрешность как среднее время, помноженное на 10 %
            value = mean_time * percent_coeff
            # погрешность в 50 мс
            value = 0.05

            random_noise = (2 * rnd.random() - 1) * value
            r.time += random_noise


def forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
            display_stat=False, visualize_res=True,
            calc_rays_p=True, calc_rays_s=True,
            calc_reflection_p=True, calc_reflection_s=True,
            calc_refraction_p=False, calc_refraction_s=False,
            noise=False):
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

    # Создание моделей геол среды и среды наблюдения (последнее из источников и приемников)
    model = SeismicModel1D(vp, vs, rho, h, phi)
    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x) for x in x_rec]
    observe = Observation(sources, receivers)

    rays_p = None
    rays_s = None

    rays_start_time = time.time()

    # Расчет кинематических параметров
    if calc_rays_p:
        disp_func('Calculating p-rays...')
        rays_p = calculate_rays(observe, model, 'vp')

        if noise:
            add_noise_rays(rays_p, model.get_depths())

    if calc_rays_s:
        disp_func('Calculating s-rays...')
        rays_s = calculate_rays(observe, model, 'vs')

        if noise:
            add_noise_rays(rays_s, model.get_depths())

    disp_func('Rays calculated!')

    reflection_start_time = time.time()

    if calc_reflection_p:
        disp_func('Calculating p-reflections...')

        calculate_reflections(model, rays_p, 'PdPu')

        disp_func('Reflections calculated!')

    if calc_reflection_s:
        disp_func('Calculating s-reflections...')

        calculate_reflections(model, rays_s, 'SdSu')

        disp_func('Reflections calculated!')

    refraction_start_time = time.time()

    if calc_refraction_p:
        disp_func('Calculating p-refractions...')

        calculate_refractions(model, rays_p, 'vp')

        disp_func('Refractions calculated!')

    if calc_refraction_s:
        disp_func('Calculating s-refractions...')

        calculate_refractions(model, rays_s, 'vs')

        disp_func('Refractions calculated!')

    calc_stop_time = time.time()

    # print('rp time = {}'.format(rays_start_time - rp_start_time))
    # print('ray tracing time = {}'.format(reflection_start_time - rays_start_time))
    # print('reflection time = {}'.format(refraction_start_time - reflection_start_time))
    # print('refraction time = {}'.format(calc_stop_time - refraction_start_time))
    # print('all_time = {}'.format(calc_stop_time - rp_start_time))

    if visualize_res:
        max_depth = model.get_max_boundary_depth() * 1.2
        dz = 100
        disp_func('Drawing results...')
        fig, axes = plt.subplots(nrows=3, ncols=2)

        # visualize_model_wellogs(axes[2, 0], model, 'vp')
        visualize_model1D(axes[2, 0], model, observe, max_depth, dz, 'vp', only_boundaries=True)
        visualize_rays_model_1D(axes[2, 0], rays_p)
        axes[2, 0].invert_yaxis()
        # axes[2, 0].set_title('model and rays for p-waves')

        # visualize_model_wellogs(axes[2, 0], model, 'vs')
        visualize_model1D(axes[2, 1], model, observe, max_depth, dz, 'vs', only_boundaries=True)
        visualize_rays_model_1D(axes[2, 1], rays_s)
        axes[2, 1].invert_yaxis()
        # axes[2, 1].set_title('model and rays for s-waves')

        visualize_time_curves(axes[1, 0], model, rays_p, observe)
        axes[1, 0].set_title('time curves for p-waves')
        visualize_time_curves(axes[1, 1], model, rays_s, observe)
        axes[1, 1].set_title('time curves for s-waves')

        if calc_reflection_p:
            visualize_reflection_amplitudes(axes[0, 0], model.get_depths()[1:], rays_p, absc='angle')
            axes[0, 0].set_title('avo for p-waves')

        if calc_reflection_s:
            visualize_reflection_amplitudes(axes[0, 1], model.get_depths()[1:], rays_s, absc='angle')
            axes[0, 1].set_title('avo for for s-waves')

        plt.show()

    return observe, model, rays_p, rays_s


def create_seismogram(seismogram, rays, observe, times, dt):
    for j, rec in enumerate(observe.receivers):
        offset = rec.x
        # rays_ = [r for r in np.nditer(rays) if abs(r.x_finish - offset) <= 0.2]
        rays_ = rays[:, j]
        trace_i = np.zeros(len(times))

        for ray in rays_:
            # ampl_curve = [r for r in reflections if float(r.boundary_z) == float(ray.reflection_z)][0]
            # r_coeff = ampl_curve.get_amplitude_by_offset(offset)
            r_coeff = ray.calculate_dynamic_factor()

            reflection_index = int(ray.time / dt)

            if reflection_index < len(trace_i):
                trace_i[reflection_index] = r_coeff.real

        signal = ricker(50, 4)
        signal /= max(signal)

        trace_values = np.convolve(trace_i, signal)[0: len(times)]

        seismogram.add_trace(Trace(trace_values, dt, offset))


def forward_with_trace_calcing(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, dt, trace_len,
                               display_stat=False, visualize_res=False,
                               use_p_waves=True, use_s_waves=True,
                               visualize_seismograms=False):
    observe, model, rays_p, rays_s = forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
                                             display_stat=display_stat, visualize_res=visualize_res,
                                             calc_rays_p=use_p_waves, calc_rays_s=use_s_waves,
                                             calc_reflection_p=use_p_waves, calc_reflection_s=use_s_waves,
                                             calc_refraction_p=use_p_waves, calc_refraction_s=use_s_waves)

    times = [dt * i for i in range(trace_len)]

    seismogram_p = None
    seismogram_s = None

    if use_p_waves:
        seismogram_p = Seismogram()
        create_seismogram(seismogram_p, rays_p, observe, times, dt)

    if use_s_waves:
        seismogram_s = Seismogram()
        create_seismogram(seismogram_s, rays_s, observe, times, dt)

    if visualize_seismograms:
        fig, axes = plt.subplots(nrows=1, ncols=2)

        if use_p_waves:
            visualize_seismogram(axes[0], seismogram_p, normalize=True, wigles=False)
            axes[0].set_title('p-waves seismogram')

        if use_s_waves:
            visualize_seismogram(axes[1], seismogram_s, normalize=True, wigles=False)
            axes[1].set_title('s-waves seismogram')

        plt.show()

    return observe, model, rays_p, rays_s, seismogram_p, seismogram_s

import numpy as np
from scipy.signal import ricker
import time
import matplotlib.pyplot as plt
import random as rnd

from ForwardModeling.RockPhysics.Models import model_calculation

from Objects.Models.Models import SeismicModel1D
from Objects.Seismic.Observation import Observation, Source, Receiver
from Objects.Seismic.Seismogram import Trace, Seismogram
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays
from ForwardModeling.Seismic.Dynamic.Reflection import calculate_reflections
from ForwardModeling.Seismic.Dynamic.Refraction import calculate_refractions
from Visualization.Seismic import visualize_model1D, visualize_rays_model_1D, \
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
            wavetypes,
            display_stat=False, visualize_res=True,
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
    :param wavetypes: Лист с типами волн для работы
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


    result_rays = {}

    for wt in wavetypes:
        disp_func(f'Calculating {wt.name}-rays...')
        result_rays[wt] = calculate_rays(observe, model, wt)

        if noise:
            add_noise_rays(result_rays[wt], model.get_depths())

        ###### REFACTOR #######

        disp_func(f'Calculating {wt.name}-reflections...')

        calculate_reflections(model, result_rays[wt], wt)

        disp_func('Calculating p-refractions...')

        calculate_refractions(model, result_rays[wt], 'vp')

        ###############


    if visualize_res:
        max_depth = model.get_max_boundary_depth() * 1.2
        dz = 100
        disp_func('Drawing results...')
        fig, axes = plt.subplots(nrows=3, ncols=len(result_rays))

        for i, key, value in enumerate(result_rays.items()):
            # visualize_model_wellogs(axes[2, 0], model, 'vp')
            ###############HARDCODE ABOUT VEL TYPE!!!!!!!!!
            visualize_model1D(axes[2, i], model, observe, max_depth, dz, 'vp', only_boundaries=True)
            visualize_rays_model_1D(axes[2, i], value)
            axes[2, i].invert_yaxis()
            # axes[2, 0].set_title('model and rays for p-waves')

            # visualize_model_wellogs(axes[2, 0], model, 'vs')
            # axes[2, 1].set_title('model and rays for s-waves')

            visualize_time_curves(axes[1, i], model, value, observe)
            axes[1, i].set_title('time curves for p-waves')

            visualize_reflection_amplitudes(axes[0, i], model.get_depths()[1:], value, absc='angle')
            axes[0, i].set_title('avo for p-waves')

        plt.show()

    return observe, model, result_rays


def create_seismogram(rays, observe, tracelen, dt):
    seismogram = Seismogram()
    times = [dt * i for i in range(tracelen)]

    for j, rec in enumerate(observe.receivers):
        offset = rec.x
        # rays_ = [r for r in np.nditer(rays) if abs(r.x_finish - offset) <= 0.2]
        rays_ = [rr for r in rays.values() for rr in r if abs(rr.x_finish - offset) <= 0.001]
        trace_i = np.zeros(len(times))

        for ray in rays_:
            # ampl_curve = [r for r in reflections if float(r.boundary_z) == float(ray.reflection_z)][0]
            # r_coeff = ampl_curve.get_amplitude_by_offset(offset)
            r_coeff = ray.calculate_dynamic_factor()

            reflection_index = int(ray.time / dt)

            if reflection_index < len(trace_i):
                trace_i[reflection_index] = r_coeff.real

        signal = ricker(50, 7)
        signal /= max(abs(signal))

        trace_values = np.convolve(trace_i, signal)[0: len(times)].real

        seismogram.add_trace(Trace(trace_values, dt, offset))

    return seismogram


def forward_with_trace_calcing(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, dt, trace_len,
                                wavetypes,
                               display_stat=False, visualize_res=False,
                               visualize_seismograms=False):
    observe, model, rays = forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
                                             wavetypes,
                                             display_stat=display_stat, visualize_res=visualize_res,)

    res_rays = {}

    for key in rays.keys():
        # TODO проверить, что сейсмограммы передаются не по ссылке!!
        seismogram = create_seismogram(rays[key], observe, dt)

        res_rays[key] = {
            "rays": rays[key],
            "seismogram": seismogram
        }

    if visualize_seismograms:
        fig, axes = plt.subplots(nrows=1, ncols=len(rays))

        for i, key in enumerate(res_rays.keys()):
            if len(rays) == 1:
                visualize_seismogram(fig, axes, res_rays[key]["seismogram"], normalize=True, wiggles=False)
                axes.set_title('waves seismogram')

            else:
                visualize_seismogram(fig, axes[i], res_rays[key]["seismogram"], normalize=True, wiggles=False)
                axes[i].set_title('waves seismogram')

        plt.show()

    return observe, model, res_rays

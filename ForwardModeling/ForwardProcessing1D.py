import numpy as np
import matplotlib.pyplot as plt

from ForwardModeling.RockPhysics.Models import simple_model_1, model_calculation
from Objects.Models.Models import SeismicModel1D
from Objects.Observation import Observation, Source, Receiver
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays
from Visualization.Seismic import visualize_model1D, visualize_rays_model_1D, visualize_time_curves


def forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, display_stat=False,
            visualize_res=True):

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

    # vp = []
    # vs = []
    # rho = []

    disp_func('Calculating rockphysics model...')

    # for i in range(nlayers):
    #     vp_, vs_, rho_ = simple_model_1(Km[i], Gm[i], Ks[i], Gs[i], Kf[i], phi[i], phi_s[i], rho_s[i], rho_f[i], rho_m[i])
    #
    #     vp.append(vp_*1000)
    #     vs.append(vs_*1000)
    #     rho.append(rho_*1000)

    vp, vs, rho = model_calculation(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m)

    disp_func('Rockphysics model calculated!')

    model = SeismicModel1D(vp, vs, rho, h)
    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x) for x in x_rec]
    observe = Observation(sources, receivers)

    disp_func('Calculating p-rays...')
    rays_p = calculate_rays(observe, model, 'vp')

    disp_func('Calculating s-rays...')
    rays_s = calculate_rays(observe, model, 'vs')

    disp_func('Rays calculated!')

    max_depth = model.get_max_boundary_depth()*1.2
    dz = 100

    if visualize_res:
        disp_func('Drawing results...')
        fig, axes = plt.subplots(nrows=2, ncols=2)

        visualize_model1D(axes[1, 0], model, max_depth, dz, 'vp')
        visualize_rays_model_1D(axes[1, 0], rays_p)
        axes[1, 0].set_title('model and rays for p-waves')

        visualize_model1D(axes[1, 1], model, max_depth, dz, 'vs')
        visualize_rays_model_1D(axes[1, 1], rays_s)
        axes[1, 1].set_title('model and rays for s-waves')

        visualize_time_curves(axes[0, 0], model, rays_p, observe)
        axes[0, 0].set_title('time curves for p-waves')
        visualize_time_curves(axes[0, 1], model, rays_s, observe)
        axes[0, 1].set_title('time curves for s-waves')

        plt.show()

    return observe, model, rays_p, rays_s

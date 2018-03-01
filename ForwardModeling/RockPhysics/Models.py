import numpy as np
from scipy import interpolate

from ForwardModeling.RockPhysics.Mediums.VoigtReussHill import voigt
from ForwardModeling.RockPhysics.Mediums.DEM import DEM
from ForwardModeling.RockPhysics.Mediums.Gassmann import Kd, Ks
from ForwardModeling.RockPhysics import Tools


def simple_model_1(Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, Vm=[1], alpha=0.1):
    '''
    Самый простой вариант модели
    :param Km: Массив модулей сжатия минералов скелета
    :param Gm: Массив модулей сдвига минералов скелета
    :param Ks: Модуль сжатия глин
    :param Gs: Модуль сдвига глин
    :param Kf: Модуль сжатия флюида
    :param phi: Пористость
    :param phi_s: Коэффициент глинистости
    :param rho_s: Плотность глин
    :param rho_f: Плотность флюида
    :param rho_m: Плотность скелета
    :param Vm: Компоненты минералов (их сумма должна быть равна 1)
    :return: Vp, Vs, Rho
    '''

    # Осреднение упругих модулей скелета
    Km_ = voigt(Km, Vm)
    Gm_ = voigt(Gm, Vm)

    # Кол-во твердого матрикса
    phi_m = 1 - phi

    # Упругие модули твердого матрикса
    Km_ = voigt(np.array([Km_, Ks]), np.array([(phi_m - phi_s) / phi_m, phi_s / phi_m]))
    Gm_ = voigt(np.array([Gm_, Gs]), np.array([(phi_m - phi_s) / phi_m, phi_s / phi_m]))

    # Создание дыр в "сухой" породе
    Km_list, Gm_list, phi_list = DEM(Km_, Gm_, np.array([0]), np.array([0]), np.array([alpha]), np.array([1]))
    Km_inter = interpolate.interp1d(Km_, phi_list)
    Gm_inter = interpolate.interp1d(Gm_, phi_list)

    K_dry = Km_inter(phi)
    G_dry = Gm_inter(phi)

    # Заливаем флюид по Гассману
    K_res = Ks(K_dry, Km_, Kf, phi)
    G_res = G_dry

    # Осредняем все плотности до результирующей
    rho_res = rho_s*phi_s + rho_f*phi + (1 - (phi + phi_s))*rho_m

    return Tools.vp_from_KGRho(K_res, G_res, rho_res), Tools.vs_from_GRho(G_res, rho_res), rho_res



import multiprocessing as mp

import numpy as np

from Exceptions.bad_calcs import BadRPModelException
from fmodeling.rock_physics import Tools
from fmodeling.rock_physics.Mediums import Gassmann
from fmodeling.rock_physics.Mediums.DEMSlb import DEM
from fmodeling.rock_physics.Mediums.VoigtReussHill import voigt, reuss
from fmodeling.rock_physics.Mediums.BGTL import bgtl

# TODO добавить векторизацию!
def xu_payne_model(Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, Vm=None, alpha=0.1):
    '''
    Модель Шу-Пэйна
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
    :param alpha: Аспектное соотношение?..
    :return: Vp, Vs, Rho
    '''

    if Vm is None:
        Vm = [1]

    # Осреднение упругих модулей скелета
    if type(Km) == np.ndarray:
        Km_ = voigt(Km, Vm)

    else:
        Km_ = Km

    if type(Gm) == np.ndarray:
        Gm_ = voigt(Gm, Vm)

    else:
        Gm_ = Gm

    # Кол-во твердого матрикса
    phi_m = 1 - phi

    # Упругие модули твердого матрикса
    Km_ = voigt(np.array([Km_, Ks]), np.array([(phi_m - phi_s) / phi_m, phi_s / phi_m]))
    Gm_ = voigt(np.array([Gm_, Gs]), np.array([(phi_m - phi_s) / phi_m, phi_s / phi_m]))

    if phi > 0:
        # Создание дыр в "сухой" породе
        K_dry, G_dry = DEM(Km_, Gm_, 0, 0, phi, alpha)

        if Kf > 0:
            # Заливаем флюид по Гассману
            K_res = Gassmann.Ks(K_dry, Km_, Kf, phi)

        else:
            K_res = K_dry

        G_res = G_dry

    else:
        K_res = Km_
        G_res = Gm_

    # Осредняем плотность скелета
    Rho_m = voigt(rho_m, Vm)

    # Осредняем все плотности до результирующей
    rho_res = rho_s*phi_s + rho_f*phi + (1 - (phi + phi_s))*Rho_m

    if K_res < 0 or G_res < 0 or rho_res < 0:
        raise BadRPModelException()

    return [Tools.vp_from_KGRho(K_res, G_res, rho_res)*1000, Tools.vs_from_GRho(G_res, rho_res)*1000, rho_res*1000]


def unconsolidated_model(Ksi, Gsi, rhosi, Ksh, Gsh, rhosh, Kincl, Gincl, rhoincl, Kfl, rhofl, Vsi, Vsh, Vincl, phi):
    """
    Модель для расчета донных песчаных осадков, заполненных газом
    :param Ksi: Модуль сжатия песка
    :param Gsi: Модуль сдвига песка
    :param rhosi: Плотность песка
    :param Ksh: Модуль сжатия глины
    :param Gsh: Модуль сдвига глины
    :param rhosh: Плотность глины
    :param Kincl: Модуль сжатия инклюзий
    :param Gincl: Модуль сдвига инклюзий
    :param rhoincl: Плотность инклюзий
    :param Kfl: Модуль сжатия флюида
    :param rhofl: Плотность флюида
    :param Vsi: Количество глины
    :param Vsh: Количество глины
    :param Vincl: Количество инклюзий
    :param phi: Пористость
    :return:
    """
    if not sum((Vsi, Vsh, Vincl, phi)) == 1:
        raise ValueError("All volumes must be 100% together!")

    Ksish = reuss(np.array((Ksi, Ksh)), np.array((Vsi / (Vsi + Vsh), Vsh / (Vsi + Vsh))))
    Gsish = reuss(np.array((Gsi, Gsh)), np.array((Vsi / (Vsi + Vsh), Vsh / (Vsi + Vsh))))
    Vsish = Vsi + Vsh

    Km = reuss(np.array((Ksish, Kincl)), np.array((Vsish/(Vsish + Vincl), Vincl/(Vsish + Vincl))))
    Gm = reuss(np.array((Gsish, Gincl)), np.array((Vsish / (Vsish + Vincl), Vincl / (Vsish + Vincl))))
    Vm = Vsish + Vincl

    Kfinal, Gfinal = bgtl(Km, Gm, Kfl, phi)

    rho_final = rhosh * Vsh + rhosi * Vsi + rhoincl * Vincl + rhofl * phi
    # проверить размерности!!!
    return [Tools.vp_from_KGRho(Kfinal, Gfinal, rho_final) * 1000, Tools.vs_from_GRho(Gfinal, rho_final) * 1000,
            rho_final * 1000]


def model_calculation_mp_helper(args):
    return xu_payne_model(*args)


def model_calculation(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, Vm=None, alpha=None, mp_cond=False):

    result_models = []

    if Vm is None:
        Vm = [1]*nlayers

    if alpha is None:
        alpha = [0.1]*nlayers

    if mp_cond:
        ncpu = mp.cpu_count()
        nproc = 1 * ncpu

        input_args = np.column_stack((Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, Vm, alpha))

        with mp.Pool(processes=nproc) as pool:
            result_models = pool.map(model_calculation_mp_helper, input_args)

    else:
        for i in range(nlayers):
            res_model = xu_payne_model(Km[i], Gm[i], Ks[i], Gs[i], Kf[i], phi[i], phi_s[i], rho_s[i], rho_f[i], rho_m[i],
                                       Vm[i], alpha[i])

            result_models.append(res_model)

    result_models = np.array(result_models)

    vp = result_models[:, 0]
    vs = result_models[:, 1]
    rho = result_models[:, 2]

    return vp, vs, rho


# TODO сделать унверсальный конструткор рокфизических сред!!
def calculate_rockphysics_model(rp_attribute):
    if rp_attribute.model_name == 'xu-payne':
        return xu_payne_model(**rp_attribute.get_input_params())

    elif rp_attribute.model_name == 'unconsolidated':
        return unconsolidated_model(**rp_attribute.get_input_params())

    else:
        raise ValueError("Unknown model name =(")

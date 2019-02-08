import numpy as np
from ForwardModeling.Seismic.Dynamic.ZoeppritzCoeffs import pdownpdown, puppup


def calculate_refraction_for_ray(model, ray, element):
    """
    Расчет коэффициентов преломления для одного луча
    :param model: Геологическая модель
    :param ray: единичный луч
    :param element: тип волны (vp или vs)
    :return:
    """

    if 's' in element.lower():
        raise NotImplementedError("S-waves are not implemented yet! GFY!")

    # Берем все границы кроме последней
    vp1_arr = model.get_param(param_name='vp', index_finish=-1)
    vs1_arr = model.get_param(param_name='vs', index_finish=-1)
    rho1_arr = model.get_param(param_name='rho', index_finish=-1)

    vp2_arr = model.get_param(param_name='vp', index_start=1)
    vs2_arr = model.get_param(param_name='vs', index_start=1)
    rho2_arr = model.get_param(param_name='rho', index_start=1)

    # Кол-во углов падения = кол-ву преломляющих границ (вот это да!)
    nangles = len(vp1_arr)

    if nangles == 0:
        return 1

    falling_angles = np.array([ray.get_boundary_angle(i) for i in range(1, nangles+1)])
    rising_angles = np.array([ray.get_boundary_angle(i) for i in range(nangles+1, 2*nangles+1)])

    # TODO чекнуть правильность определения углов падения + чекнуть, правильно ли считаются к-ты прохождения
    down_coeffs = pdownpdown(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, falling_angles)
    up_coeffs = puppup(vp2_arr[::-1], vs2_arr[::-1], rho2_arr[::-1], vp1_arr[::-1], vs1_arr[::-1], rho1_arr[::-1], rising_angles)

    refr_coeff = np.prod(np.append(down_coeffs, up_coeffs))

    return refr_coeff


def calculate_refraction(model, rays, element):
    refractions = []
    for ray in rays:
        refractions.append(calculate_refraction_for_ray(model, ray, element))

    return refractions

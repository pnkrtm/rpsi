import numpy as np
from ForwardModeling.Seismic.Dynamic.ZoeppritzCoeffs import pdownpdown, puppup
from Objects.Seismic.Rays import BoundaryType


def calculate_refraction_for_ray(model, ray, element):
    """
    Расчет коэффициентов преломления для одного луча
    :param model: Геологическая модель
    :param ray: единичный луч
    :param element: тип волны (vp или vs). Но сейчас прилетает PdPu
    :return:
    """

    if 's' in element.lower():
        raise NotImplementedError("S-waves are not implemented yet! GFY!")

    nrefractions = ray.nlayers - 1
    npoints = ray.nlayers * 2 + 1

    if nrefractions == 0:
        return

    # Берем все границы кроме последней
    vp1_arr = model.get_param(param_name='vp', index_finish=nrefractions)
    vs1_arr = model.get_param(param_name='vs', index_finish=nrefractions)
    rho1_arr = model.get_param(param_name='rho', index_finish=nrefractions)

    vp2_arr = model.get_param(param_name='vp', index_start=1, index_finish=nrefractions+1)
    vs2_arr = model.get_param(param_name='vs', index_start=1, index_finish=nrefractions+1)
    rho2_arr = model.get_param(param_name='rho', index_start=1, index_finish=nrefractions+1)

    # Кол-во углов падения = кол-ву преломляющих границ (вот это да!)
    nangles = nrefractions

    falling_angles = np.array([ray.get_boundary_angle(i) for i in range(1, nangles+1)])
    rising_angles = np.array([ray.get_boundary_angle(i) for i in range(npoints - 1 - nangles, npoints - 1)])[::-1]

    # TODO восходящие к-ты неправильные
    down_coeffs = pdownpdown(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, falling_angles)
    up_coeffs = puppup(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, rising_angles)

    return down_coeffs, up_coeffs


# TODO сделать расчет к-тов прохождения не по лучам, а по границам
def calculate_refractions(model, rays, element):
    # rays[0] - это отражения от первой границы, у которых нет к-тов преломления
    for rays_depth in rays[1:]:
        for ray in rays_depth:
            down_coeffs, up_coeffs = calculate_refraction_for_ray(model, ray, element)

            i = 1
            for dc, uc in zip(down_coeffs, up_coeffs):
                ray.add_boundary_dynamic(dc, BoundaryType.REFRACTION_DOWN, i)
                ray.add_boundary_dynamic(uc, BoundaryType.REFRACTION_UP, i)

                i += 1

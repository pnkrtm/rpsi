import numpy as np
from ForwardModeling.Seismic.Dynamic.ZoeppritzCoeffs import pdownpdown, puppup, svdownsvdown, svupsvup
from Objects.Seismic.Rays import BoundaryType
from Objects.Data.RDPair import OWT


def calculate_refraction_for_ray(model, ray, owt):
    """
    Расчет коэффициентов преломления для одного луча
    :param model: Геологическая модель
    :param ray: единичный луч
    :param owt: Observation wavetype.
    :return:
    """

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

    if owt == OWT.PdPu:
        down_coeffs = pdownpdown(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, falling_angles)
        up_coeffs = puppup(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, rising_angles)

    elif owt == owt.SVdSVu:
        down_coeffs = svdownsvdown(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, falling_angles)
        up_coeffs = svupsvup(vp1_arr, vs1_arr, rho1_arr, vp2_arr, vs2_arr, rho2_arr, rising_angles)

    return down_coeffs, up_coeffs


# TODO сделать расчет к-тов прохождения не по лучам, а по границам
def calculate_refractions(model, rays, wtype):
    # rays[0] - это отражения от первой границы, у которых нет к-тов преломления
    depths = model.get_depths()
    reflection_indexes = np.array(list(rays.keys()))

    # for i, d in enumerate(depths[1: ], 1):
    #     target_rays = []
    #
    #     indxtmp = reflection_indexes[reflection_indexes > i]
    #     for idx in indxtmp:
    #         target_rays = np.append(target_rays, rays[idx])

    for bound_ind, rays_depth in rays.items():
        if bound_ind > 1:

            for ray in rays_depth:
                down_coeffs, up_coeffs = calculate_refraction_for_ray(model, ray, wtype)

                i = 1
                for dc, uc in zip(down_coeffs, up_coeffs):
                    ray.add_boundary_dynamic(dc, BoundaryType.REFRACTION_DOWN, i)
                    ray.add_boundary_dynamic(uc, BoundaryType.REFRACTION_UP, i)

                    i += 1

import numpy as np
from fmodeling.seismic.dynamic.zoeppritz_coeffs import pdownpdown, puppup, svdownsvdown, svupsvup
from fmodeling.seismic.dynamic.zoeppritz_coeffs_water import pdownpdown_water as pdpd_sea, puppup_water as pupu_sea
from objects.seismic.rays import BoundaryType
from objects.seismic.waves import OWT


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
    vp1_arr = model.get_single_param(param_name='vp', index_finish=nrefractions)
    vs1_arr = model.get_single_param(param_name='vs', index_finish=nrefractions)
    rho1_arr = model.get_single_param(param_name='rho', index_finish=nrefractions)

    vp2_arr = model.get_single_param(param_name='vp', index_start=1, index_finish=nrefractions + 1)
    vs2_arr = model.get_single_param(param_name='vs', index_start=1, index_finish=nrefractions + 1)
    rho2_arr = model.get_single_param(param_name='rho', index_start=1, index_finish=nrefractions + 1)

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

    else:
        raise NotImplementedError(f"Refraction type {owt} is not implemented yet!")

    return down_coeffs, up_coeffs

# TODO проверить правильность расчета углов и соответствующих к-тов прохождения!!
def calculate_refraction_vectorized(model, rays, owt):
    depths = model.get_depths()
    reflection_indexes = np.array(list(rays.keys()))

    vp1_list, vs1_list, rho1_list = [], [], []
    vp2_list, vs2_list, rho2_list = [], [], []
    falling_angles_list = []
    rising_angles_list = []
    rays_indexes_1 = []
    rays_indexes_2 = []
    boundary_indexes = []

    for i, d in enumerate(depths[1:-1], 1):
        vp1_tmp, vs1_tmp, rho1_tmp = model.get_multiple_params(param_names=['vp', 'vs', 'rho'], index_start=i-1,
                                                                  index_finish=i)
        vp2_tmp, vs2_tmp, rho2_tmp = model.get_multiple_params(param_names=['vp', 'vs', 'rho'], index_start=i,
                                                                  index_finish=i + 1)

        refl_indx_tmp = reflection_indexes[reflection_indexes > i]

        falling_angles_tmp = [ray.get_boundary_angle(i) for idx in refl_indx_tmp for ray in rays[idx]]
        # TODO check rising angles correct!!
        rising_angles_tmp = [ray.get_boundary_angle(-i-1) for idx in refl_indx_tmp for ray in rays[idx]]

        assert (len(falling_angles_tmp) == len(rising_angles_tmp))

        falling_angles_list = np.append(falling_angles_list, falling_angles_tmp)
        rising_angles_list = np.append(rising_angles_list, rising_angles_tmp)

        nrays = len(falling_angles_tmp)

        rays_indexes_1_tmp = [[idx]*len(rays[idx]) for idx in refl_indx_tmp]
        rays_indexes_2_tmp = [np.arange(len(rays[idx])) for idx in refl_indx_tmp]

        vp1_list = np.append(vp1_list, [vp1_tmp] * nrays)
        vs1_list = np.append(vs1_list, [vs1_tmp] * nrays)
        rho1_list = np.append(rho1_list, [rho1_tmp] * nrays)

        vp2_list = np.append(vp2_list, [vp2_tmp] * nrays)
        vs2_list = np.append(vs2_list, [vs2_tmp] * nrays)
        rho2_list = np.append(rho2_list, [rho2_tmp] * nrays)

        boundary_indexes = np.append(boundary_indexes, [i] * nrays)

        rays_indexes_1 = np.append(rays_indexes_1, rays_indexes_1_tmp)
        rays_indexes_2 = np.append(rays_indexes_2, rays_indexes_2_tmp)

    if owt == OWT.PdPu:
        down_coeffs = pdownpdown(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, falling_angles_list)
        up_coeffs = puppup(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, rising_angles_list)

    elif owt == owt.SVdSVu:
        down_coeffs = svdownsvdown(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, falling_angles_list)
        up_coeffs = svupsvup(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, rising_angles_list)

    elif owt == OWT.PdSVu:
        down_coeffs = pdownpdown(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, falling_angles_list)
        up_coeffs = svupsvup(vp1_list, vs1_list, rho1_list, vp2_list, vs2_list, rho2_list, rising_angles_list)

    elif owt == owt.PdPu_water:
        # кол-во
        nwaterrays = sum(rays_indexes_1 == 2)
        down_coeffs_bottom = pdpd_sea(vp1_list[0: 1], rho1_list[0: 1], vp2_list[0: 1], vs2_list[0: 1], rho2_list[0: 1], falling_angles_list[0: 1])
        up_coeffs_bottom = pupu_sea(vp1_list[0: 1], rho1_list[0: 1], vp2_list[0: 1], vs2_list[0: 1], rho2_list[0: 1], falling_angles_list[0: 1])

        down_coeffs_underbottom = pdownpdown(vp1_list[1:], vs1_list[1:], rho1_list[1:], vp2_list[1:], vs2_list[1:], rho2_list[1:], rising_angles_list[1:])
        up_coeffs_underbottom = puppup(vp1_list[1:], vs1_list[1:], rho1_list[1:], vp2_list[1:], vs2_list[1:], rho2_list[1:], rising_angles_list[1:])

        down_coeffs = np.concatenate((down_coeffs_bottom, down_coeffs_underbottom))
        up_coeffs = np.concatenate((up_coeffs_bottom, up_coeffs_underbottom))

    else:
        raise ValueError(f"Refraction type {owt} is not implemented yet!")

    for dc, uc, ri_1, ri_2, bi in zip(down_coeffs, up_coeffs, rays_indexes_1, rays_indexes_2, boundary_indexes):
        rays[int(ri_1)][int(ri_2)].set_boundary_dynamic(dc, BoundaryType.REFRACTION_DOWN, bi, depths[int(bi)])


# TODO сделать расчет к-тов прохождения не по лучам, а по границам
def calculate_refractions(model, rays, wtype):
    # rays[0] - это отражения от первой границы, у которых нет к-тов преломления

    for bound_ind, rays_depth in rays.items():
        if bound_ind > 1:

            for ray in rays_depth:
                down_coeffs, up_coeffs = calculate_refraction_for_ray(model, ray, wtype)

                i = 1
                for dc, uc in zip(down_coeffs, up_coeffs):
                    ray.set_boundary_dynamic(dc, BoundaryType.REFRACTION_DOWN, i)
                    ray.set_boundary_dynamic(uc, BoundaryType.REFRACTION_UP, i)

                    i += 1

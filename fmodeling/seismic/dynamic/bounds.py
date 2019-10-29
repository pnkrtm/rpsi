import numpy as np


def calculate_bounds(model, rays):
    """

    :param model:
    :param rays: dictionary {boundary_index: reflected rays}
    :return:
    """

    # делаем развертку по всем границам для каждого луча для каждой границы
    bound_types = []
    angles = []
    ray_indexes = []
    # индексы границ
    bound_indexes = []

    # развертываем словарь лучей в один длинный список лучей
    rays_ = np.concatenate(list(rays.values()))
    rindexes = np.arange(len(rays_))

    # boundaries parameters
    # должен быть массив словарей все массивы должны быть одной длины (вроде как)
    # также все массивы должны быть друг с другом синхронизированы
    # и одинаковой длины
    ray_indexes = np.concatenate([[ri] * len(r.boundaries) for ri, r in zip(rindexes, rays_)])
    boundaries = np.concatenate([list(r.boundaries.values()) for r in rays_])
    indexes = np.concatenate([list(r.boundaries.keys()) for r in rays_])
    bound_indexes = [b["boundary_index"] for b in boundaries]
    bound_types = np.array([b["boundary_type"].__class__ for b in boundaries])
    angles = np.array([rays_[ri].get_boundary_angle(i) for ri, i in zip(ray_indexes, indexes)])

    assert len(ray_indexes) == len(boundaries) == len(indexes) == len(bound_indexes) == len(bound_types) == len(angles)

    boundary_coeffs = np.array([np.NAN]*len(boundaries))

    unique_bounds = list(set(bound_types))

    vp1_arr = np.array([model.get_single_param(param_name='vp', index_start=bi - 1, index_finish=bi)[0] for bi in
               bound_indexes])
    vs1_arr = np.array([model.get_single_param(param_name='vs', index_start=bi - 1, index_finish=bi)[0] for bi in
               bound_indexes])
    rho1_arr = np.array([model.get_single_param(param_name='rho', index_start=bi - 1, index_finish=bi)[0] for bi in
                bound_indexes])

    vp2_arr = np.array([model.get_single_param(param_name='vp', index_start=bi, index_finish=bi + 1)[0] for bi in
               bound_indexes])
    vs2_arr = np.array([model.get_single_param(param_name='vs', index_start=bi, index_finish=bi + 1)[0] for bi in
               bound_indexes])
    rho2_arr = np.array([model.get_single_param(param_name='rho', index_start=bi, index_finish=bi + 1)[0] for bi in
                bound_indexes])

    for ub in unique_bounds:
        idx = bound_types == ub
        # TODO проверить правильность индексов границ и слоев
        # чекнуть, что к-ты отражения правильно расставляются по своим местам

        boundary_coeffs[idx] = ub.calculate_coeff(vp1_arr[idx], vs1_arr[idx], rho1_arr[idx],
                                                  vp2_arr[idx], vs2_arr[idx], rho2_arr[idx],
                                                  angles[idx])

    for i, bc in enumerate(boundary_coeffs):
        rays_[ray_indexes[i]].set_boundary_dynamic(indexes[i], bc)

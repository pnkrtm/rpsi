from objects.seismic.waves import OWT
import numpy as np
from objects.seismic.boundaries.refl_boundary_up import PDownPUpReflection, PDownPUpWaterReflection
from objects.seismic.boundaries.trans_boundary_down import PDownPDownTransmission, PDownPDownWaterTransmission
from objects.seismic.boundaries.trans_boundary_up import PUpPUpTransmission, PUpPUpWaterTransmission


def PdPu(indexes, bound_indexes, refl_index):
    if len(indexes) > 1:
        down_bounds = [PDownPDownTransmission(i) for i in indexes[bound_indexes < refl_index]]
        up_bounds = [PUpPUpTransmission(i) for i in indexes[bound_indexes > refl_index]]

        bounds = np.concatenate((down_bounds, [PDownPUpReflection(refl_index)], up_bounds))

    else:
        bounds = [PDownPUpReflection(refl_index)]

    return bounds


def PdPuWater(indexes, bound_indexes, refl_index):
    if len(indexes) > 1:
        down_bounds_1 = [PDownPDownWaterTransmission(1)]
        down_bounds_2 = [PDownPDownTransmission(i) for i in indexes[(bound_indexes < refl_index) & (bound_indexes > 1)]]

        up_bounds_2 = [PUpPUpTransmission(i) for i in indexes[(bound_indexes > refl_index) & (bound_indexes < bound_indexes[-1])]]
        up_bounds_1 = [PUpPUpWaterTransmission(indexes[-1])]

        bounds = np.concatenate((down_bounds_1, down_bounds_2, [PDownPUpReflection(refl_index)], up_bounds_2, up_bounds_1))

    else:
        bounds = [PDownPUpWaterReflection(refl_index)]

    return bounds


def strategy_1D(z_points, owt):
    assert len(z_points) % 2 == 1

    refl_index = int(len(z_points) / 2)

    bound_indexes = [i for i in range(1, refl_index+1)]

    if len(bound_indexes) > 1:
        bound_indexes = np.append(bound_indexes, bound_indexes[::-1][1:])

    indexes = np.arange(1, len(z_points) - 1)

    if owt == OWT.PdPu:
        bounds = PdPu(bound_indexes, indexes, refl_index)

    elif owt == OWT.PdPu_water:
        bounds = PdPuWater(bound_indexes, indexes, refl_index)

    else:
        raise NotImplementedError(f"Strategy {owt} is not implemented yet")

    # TODO проверить, что длина массива с индексами совпадает с длиной массива с границами
    return z_points[1:-1], indexes, bound_indexes, bounds
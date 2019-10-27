from typing import Union
import numpy as np

from fmodeling.Utils.classes_keeper import ClassesKeeper

boundary_types = ClassesKeeper()


def register_boundary(K):  # ToDo check names collisions
    boundary_types.add_class(K, None)
    return K


class BaseBoundary:
    @property
    def wave_direction_1(self):
        """
        First wave's direction on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_direction_2(self):
        """
        Second wave's direction on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_type_1(self):
        """
        First wave's type on boundary
        :return:
        """
        raise NotImplementedError()

    @property
    def wave_type_2(self):
        """
        First wave's type on boundary
        :return:
        """
        raise NotImplementedError()

    def calculate_coeff(self, vp1: Union[float, np.ndarray], vs1: Union[float, np.ndarray], rho1: Union[float, np.ndarray],
                            vp2: Union[float, np.ndarray], vs2: Union[float, np.ndarray], rho2: Union[float, np.ndarray],
                        theta1: Union[float, np.ndarray] = 0):
        raise NotImplementedError()

from typing import Union

import numpy as np

from fmodelling.seismic.dynamic.zoeppritz_coeffs import pdownpup
from fmodelling.seismic.dynamic.zoeppritz_coeffs_water import pdownpup_water
from objects.seismic.boundaries.base_boundary import BaseBoundary
from objects.seismic.boundaries.base_boundary import register_boundary
from objects.seismic.waves import WD, WT


class ReflectionUp(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.DOWN

    @property
    def wave_direction_2(self):
        return WD.UP


class PDownPUpReflection(ReflectionUp):
    @property
    def wave_type_1(self):
        return WT.P

    @property
    def wave_type_2(self):
        return WT.P

    @classmethod
    def calculate_coeff(cls, vp1: Union[float, np.ndarray], vs1: Union[float, np.ndarray], rho1: Union[float, np.ndarray],
                            vp2: Union[float, np.ndarray], vs2: Union[float, np.ndarray], rho2: Union[float, np.ndarray],
                        theta1: Union[float, np.ndarray] = 0):
        return pdownpup(vp1, vs1, rho1, vp2, vs2, rho2, theta1)


@register_boundary
class PDownPUpWaterReflection(ReflectionUp):
    @property
    def wave_type_1(self):
        return WT.P

    @property
    def wave_type_2(self):
        return WT.P

    @classmethod
    def calculate_coeff(cls, vp1: Union[float, np.ndarray], vs1: Union[float, np.ndarray],
                        rho1: Union[float, np.ndarray],
                        vp2: Union[float, np.ndarray], vs2: Union[float, np.ndarray], rho2: Union[float, np.ndarray],
                        theta1: Union[float, np.ndarray] = 0):
        return pdownpup_water(vp1, rho1, vp2, vs2, rho2, theta1)

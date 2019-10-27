from typing import Union

import numpy as np

from fmodeling.seismic.dynamic.zoeppritz_coeffs import pdownpdown
from fmodeling.seismic.dynamic.zoeppritz_coeffs_water import pdownpdown_water
from objects.seismic.boundaries.base_boundary import BaseBoundary
from objects.seismic.waves import WD, WT


class TransmissionDown(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.DOWN

    @property
    def wave_direction_2(self):
        return WD.DOWN


class PDownPDownTransmission(TransmissionDown):
    @property
    def wave_type_1(self):
        return WT.P

    @property
    def wave_type_2(self):
        return WT.P

    def calculate_coeff(self, vp1: Union[float, np.ndarray], vs1: Union[float, np.ndarray], rho1: Union[float, np.ndarray],
                            vp2: Union[float, np.ndarray], vs2: Union[float, np.ndarray], rho2: Union[float, np.ndarray],
                        theta1: Union[float, np.ndarray] = 0):
        return pdownpdown(vp1, vs1, rho1, vp2, vs2, rho2, theta1)


class PDownPDownWaterTransmission(TransmissionDown):
    @property
    def wave_type_1(self):
        return WT.P

    @property
    def wave_type_2(self):
        return WT.P

    def calculate_coeff(self, vp1: Union[float, np.ndarray], vs1: Union[float, np.ndarray],
                        rho1: Union[float, np.ndarray],
                        vp2: Union[float, np.ndarray], vs2: Union[float, np.ndarray], rho2: Union[float, np.ndarray],
                        theta1: Union[float, np.ndarray] = 0):
        return pdownpdown_water(vp1, rho1, vp2, vs2, rho2, theta1)


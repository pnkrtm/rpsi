from typing import Union

import numpy as np

from fmodelling.seismic.dynamic.zoeppritz_coeffs import puppup
from fmodelling.seismic.dynamic.zoeppritz_coeffs_water import puppup_water
from objects.seismic.boundaries.base_boundary import BaseBoundary
from objects.seismic.waves import WD, WT


class TransmissionUp(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.UP

    @property
    def wave_direction_2(self):
        return WD.UP


class PUpPUpTransmission(TransmissionUp):
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
        return puppup(vp1, vs1, rho1, vp2, vs2, rho2, theta1)


class PUpPUpWaterTransmission(TransmissionUp):
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
        return puppup_water(vp1, rho1, vp2, vs2, rho2, theta1)

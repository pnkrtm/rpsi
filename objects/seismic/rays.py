from enum import Enum

import numpy as np
from objects.seismic.waves import OWT
from collections import defaultdict


class Ray:
    def __init__(self):
        self.p = -1
        self.boundaries = {}

    @property
    def x_start(self):
        raise NotImplementedError()

    @property
    def x_finish(self):
        raise NotImplementedError()

    @property
    def reflection_z(self):
        raise NotImplementedError

    def create_boundaries(self, depths, indexes, types):
        self.boundaries = {idx: {
            "boundary_depth": dpth,
            "boundary_type": t,
            "coeff": -1
        } for idx , dpth, t in zip(indexes, depths, types)}


    def set_boundary_dynamic(self, index, value):
        if abs(value) > 1:
            value = 1 + np.exp(-abs(value))

        self.boundaries[index]["coeff"] = value

    def calculate_dynamic_factor(self):
        vals = [b["coeff"] for b in self.boundaries]

        return np.prod(vals)


class Ray1D(Ray):
    def __init__(self, dutype, x_points=None, z_points=None, time=-1, p=-1, offset=-1):
        """

        :param dutype: Down-Up type (enumerator ObservationDataTypes)
        :param x_points:
        :param z_points:
        :param time:
        :param p:
        :param offset:
        """
        super().__init__()

        self._dutype = dutype

        if x_points is not None:
            self.x_points = x_points
        else:
            self.x_points = []

        if z_points is not None:
            self.z_points = z_points
        else:
            self.z_points = []

        self.offset = offset
        self.time = time
        self.p = p

    @property
    def x_start(self):
        return self.x_points[0]

    @property
    def x_finish(self):
        return self.x_points[-1]

    @property
    def reflection_z(self):
        return max(self.z_points)

    @property
    def dutype(self):
        return self._dutype

    @property
    def nlayers(self):
        return len(self.x_points) // 2

    # TODO check this shit
    def get_boundary_angle(self, bound_index):
        """
        На вход даем индекс границы (по всей видимости начиная с 1), на выходе имеем угол падения на эту границу
        :param bound_index:
        :return:
        """
        angle = np.arctan(abs((self.x_points[bound_index] - self.x_points[bound_index - 1]) /
                              (self.z_points[bound_index] - self.z_points[bound_index - 1])))

        angle = np.rad2deg(angle)

        return angle

    def get_all_boundaries_angles(self):
        angles = [self.get_boundary_angle(bi) for bi in range(1, len(self.x_points))]

        return angles

    def get_reflection_angle(self):
        """

        :return: angle in degrees!!
        """
        bound_index = len(self.x_points) // 2 + 1

        return self.get_boundary_angle(bound_index)

    def get_boundary_type(self, bound_index):
        return self.boundaries[bound_index]["boundary_type"]

    def get_all_boundaries_types(self):
        return [self.get_boundary_type(bi) for bi in self.boundaries.keys()]

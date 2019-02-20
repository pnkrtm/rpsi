from enum import Enum

import numpy as np


class BoundaryType(Enum):
    REFLECTION = 1
    REFRACTION_DOWN = 2
    REFRACTION_UP = 3


class Ray():
    def __init__(self):
        self.p = -1
        self.boundaries_dynamics = []

    @property
    def x_start(self):
        raise NotImplementedError()

    @property
    def x_finish(self):
        raise NotImplementedError()

    @property
    def reflection_z(self):
        raise NotImplementedError

    def add_boundary_dynamic(self, value, bound_type: BoundaryType, bound_index: int=-1, bound_depth: float=-1):
        self.boundaries_dynamics.append(
            {
                "boundary_index": bound_index,
                "boundadry_depth": bound_depth,
                "boundary_type": bound_type,
                "value": value
            }
        )

    def calculate_dynamic_factor(self):
        vals = [bd["value"] for bd in self.boundaries_dynamics]

        return np.prod(vals)


class Ray1D(Ray):
    def __init__(self, x_points=[], z_points=[], time=-1, p=-1, offset=-1):
        super().__init__()

        self.x_points = x_points
        self.z_points = z_points
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

    def get_reflection_angle(self):
        bound_index = len(self.x_points) // 2 + 1

        return self.get_boundary_angle(bound_index)

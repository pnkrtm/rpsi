import numpy as np

class Ray():
    def __init__(self):
        self.p = -1

    def get_x_start(self):
        pass

    def get_x_finish(self):
        pass


class Ray1D(Ray):
    def __init__(self, x_points=[], z_points=[], time=-1, p=-1, offset=-1):
        super().__init__()

        self.x_points = x_points
        self.z_points = z_points
        self.offset = offset
        self.time = time
        self.p = p

    def get_x_start(self):
        return self.x_points[0]

    def get_x_finish(self):
        return self.x_points[-1]

    def get_reflection_z(self):
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

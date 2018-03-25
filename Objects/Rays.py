class Ray():
    def __init__(self):
        self.p = -1

    def get_x_start(self):
        pass

    def get_x_finish(self):
        pass


class Ray1D(Ray):
    def __init__(self, x_points=[], z_points=[], time=-1, p=-1):
        super().__init__()

        self.x_points = x_points
        self.z_points = z_points
        self.time = time
        self.p = p

    def get_x_start(self):
        return self.x_points[0]

    def get_x_finish(self):
        return self.x_points[-1]

    def get_reflection_z(self):
        return max(self.z_points)
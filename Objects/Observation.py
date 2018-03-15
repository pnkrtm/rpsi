class Receiver:
    def __init__(self, x=-1, y=-1, z=-1):
        self.x = x
        self.y = y
        self.z = z


class Source:
    def __init__(self, x=-1, y=-1, z=-1):
        self.x = x
        self.y = y
        self.z = z


class Observation:
    def __init__(self, sources=None, receivers=None):
        self.sources = sources
        self.receivers = receivers

    def get_x_geometry(self):
        return [rec.x for rec in self.receivers]

import numpy as np

class ReflectionCurve:
    def __init__(self, amplitudes=None, offsets=None, angles=None, boundary_z=-1):
        if amplitudes is not None:
            self.amplitudes = amplitudes

        else:
            self.amplitudes = []

        self.offsets = offsets or []
        self.angles = angles or []
        self.boundary_z = boundary_z

    @property
    def doffset(self):
        return np.round(self.offsets[1] - self.offsets[0])

    def get_amplitude_by_offset(self, offset):
        # nearest_offset = min(self.offsets, key=lambda x: abs(x - offset))

        index = int(offset / self.doffset) - 1

        return self.amplitudes[index]


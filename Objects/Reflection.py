import numpy as np

class ReflectionCurve:
    def __init__(self, amplitudes=[], offsets=[], angles=[], boundary_z=-1):
        self.amplitudes = amplitudes
        self.offsets = offsets
        self.angles = angles
        self.boundary_z = boundary_z

    @property
    def doffset(self):
        return np.round(self.offsets[1] - self.offsets[0])

    def get_amplitude_by_offset(self, offset):
        # nearest_offset = min(self.offsets, key=lambda x: abs(x - offset))

        index = int(offset / self.doffset) - 1

        return self.amplitudes[index]


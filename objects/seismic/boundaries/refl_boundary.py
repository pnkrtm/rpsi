from objects.seismic.boundaries.base_boundary import BaseBoundary
from objects.seismic.waves import WD


class ReflectionUp(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.DOWN

    @property
    def wave_direction_2(self):
        return WD.UP


class ReflectionDown(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.UP

    @property
    def wave_direction_2(self):
        return WD.DOWN

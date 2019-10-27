from objects.seismic.boundaries.base_boundary import BaseBoundary
from objects.seismic.waves import WD


class TransmissionDown(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.DOWN

    @property
    def wave_direction_2(self):
        return WD.DOWN


class TransmissionUp(BaseBoundary):
    @property
    def wave_direction_1(self):
        return WD.UP

    @property
    def wave_direction_2(self):
        return WD.UP

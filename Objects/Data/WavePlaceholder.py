from enum import Enum
import numpy as np


class OWT(Enum):
    """
    Observation Wave Types
    """
    PdPu = 0
    PdSVu = 1
    SVdPu = 2
    SVdSVu = 3
    SHdSHu = 4


def get_down_up_vel_types(odt):
    res = {}
    if odt in (OWT.PdPu, OWT.PdSVu):
        res['down'] = 'vp'

    else:
        res['down'] = 'vs'

    if odt in (OWT.PdPu, OWT.SVdPu):
        res['up'] = 'vp'

    else:
        res['up'] = 'vs'

    return res


class WaveDataPlaceholder:
    """
    Класс для хранения и использования пары лучи/сейсмограмма
    """
    def __init__(self, wavetype, rays, seismogram, start_indexes=None, stop_indexes=None, trace_weights=None):

        self.rays = rays
        self.seismogram = seismogram
        self._dutype = wavetype

        self.start_indexes = start_indexes
        if self.start_indexes is None:
            self.start_indexes = [0] * self.seismogram.ntraces

        self.stop_indexes = stop_indexes
        if self.stop_indexes is None:
            self.stop_indexes = [-1] * self.seismogram.ntraces

        self.trace_weights = trace_weights
        if self.trace_weights is None:
            self.trace_weights = [1 / self.seismogram.ntraces] * self.seismogram.ntraces





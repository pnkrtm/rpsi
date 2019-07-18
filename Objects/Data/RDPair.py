from enum import Enum
import numpy as np


class ODT(Enum):
    """
    Observation Data Types
    """
    PdPu = 0
    PdSVu = 1
    SVdPu = 2
    SVdSVu = 3
    SHdSHu = 4


def get_down_up_vel_types(odt):
    res = {}
    if odt in (ODT.PdPu, ODT.PdSVu):
        res['down'] = 'vp'

    else:
        res['down'] = 'vs'

    if odt in (ODT.PdPu, ODT.SVdPu):
        res['up'] = 'vp'

    else:
        res['up'] = 'vs'

    return res


def validate_all_rays(rays):
    res = [r.dutype == rr for rr in rays for r in rays]

    return False not in res


class RayDataPair:
    """
    Класс для хранения и использования пары лучи/сейсмограмма
    """
    def __init__(self, rays, seismogram):

        if validate_all_rays(rays):
            self.rays = rays
            self.seismogram = seismogram
            self._dutype = self.rays[0].dutype





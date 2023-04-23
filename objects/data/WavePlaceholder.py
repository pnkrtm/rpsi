from objects.seismic.waves import OWT


def get_down_up_vel_types(odt):
    res = {}
    if odt in (OWT.PdPu, OWT.PdSVu, OWT.PdPu_water):
        res['down'] = 'vp'

    elif odt in (OWT.SVdPu, OWT.SVdSVu, OWT.SHdSHu):
        res['down'] = 'vs'

    else:
        raise ValueError(f"Unknown rays type {odt} for down velovity")

    if odt in (OWT.PdPu, OWT.SVdPu, OWT.PdPu_water):
        res['up'] = 'vp'

    elif odt in (OWT.PdSVu, OWT.SVdSVu, OWT.SHdSHu):
        res['up'] = 'vs'

    else:
        raise ValueError(f"Unknown rays type {odt} for up velovity")

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





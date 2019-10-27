from enum import Enum, auto


class OWT(Enum):
    """
    Observation Wave Types
    """
    PdPu = auto()
    PdSVu = auto()
    SVdPu = auto()
    SVdSVu = auto()
    SHdSHu = auto()

    PdPu_water = auto()


class WD(Enum):
    """
    Wave Direction enum
    """
    DOWN = auto()
    UP = auto()


class WT(Enum):
    """
    Wave Type
    """
    P = auto()
    SV = auto()
    SH = auto()

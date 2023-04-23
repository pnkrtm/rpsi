from enum import Enum
from fmodelling.rock_physics.Models import calculate_rockphysics_model
from objects.attributes.seismic.SeismicAttribute import SeismicAttribute

class LayerOPT(Enum):
    NO = 0
    RP = 1
    SEISM = 2


class Layer1D:
    def __init__(self, h, rp_attribute=None, seism_attribute=None, refl_flag=1, opt=LayerOPT.NO):
        self.h = h
        self.rp_attribute = rp_attribute
        self.seism_attribute = seism_attribute
        self.refl_flag = refl_flag
        self.opt = opt

    def __getitem__(self, item):
        if item.lower() == 'h':
            return self.h

        elif item.lower() in ('vp', 'vs', 'rho', 'aip', 'ais'):
            return self.seism_attribute[item]

        else:
            return self.rp_attribute[item]

    def __setitem__(self, key, value):
        if key.lower() == 'h':
            self.h = value

        elif key.lower() in ('vp', 'vs', 'rho'):
            self.seism_attribute[key] = value

        else:
            self.rp_attribute[key] = value

    @property
    def is_optimization(self):
        if self.opt == LayerOPT.NO:
            return False

        else:
            return True

    def get_optimization_params(self):
        if self.opt == LayerOPT.RP:
            return self.rp_attribute.get_params_to_optimize()

        else:
            raise NotImplementedError()

    def set_optimized_params(self, values):
        if self.opt == LayerOPT.RP:
            self.rp_attribute.set_optimized_params(values)

        else:
            raise NotImplementedError()

    def get_optimization_min(self):
        if self.opt == LayerOPT.RP:
            return self.rp_attribute.get_min_optimize_bound()

        else:
            raise NotImplementedError()

    def get_optimization_max(self):
        if self.opt == LayerOPT.RP:
            return self.rp_attribute.get_max_optimize_bound()

        else:
            raise NotImplementedError

    def set_seismic(self, vp, vs, rho):
        self.seism_attribute.vp = vp
        self.seism_attribute.vs = vs
        self.seism_attribute.rho = rho

    def calculate_rockphysics(self):
        if self.rp_attribute is not None:
            vp, vs, rho = calculate_rockphysics_model(self.rp_attribute)

            if self.seism_attribute is None:
                self.seism_attribute = SeismicAttribute(vp, vs, rho)
            else:
                self.set_seismic(vp, vs, rho)

    @property
    def vp(self):
        return self.seism_attribute.vp

    @property
    def vs(self):
        return self.seism_attribute.vs

    @property
    def rho(self):
        return self.seism_attribute.rho
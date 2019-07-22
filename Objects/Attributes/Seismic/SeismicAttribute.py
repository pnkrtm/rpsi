from collections import OrderedDict
from Objects.Attributes.AbstractAttribute import AbstarctAttribute


class SeismicAttribute:
    def __init__(self, vp, vs, rho):
        self.vals_dict = {
            "vp": vp,
            "vs": vs,
            "rho": rho
        }

    def __getitem__(self, item):
        if item.lower() == "aip":
            return self.vals_dict["vp"] * self.vals_dict["rho"]
        elif item.lower() == "ais":
            return self.vals_dict["vs"] * self.vals_dict["rho"]
        else:
            return self.vals_dict[item]

    def __setitem__(self, key, value):
        self.vals_dict[key] = value

    @property
    def vp(self):
        return self.vals_dict["vp"]

    @vp.setter
    def vp(self, value):
        self.vals_dict["vp"] = value

    @property
    def vs(self):
        return self.vals_dict["vs"]

    @vs.setter
    def vs(self, value):
        self.vals_dict["vs"] = value

    @property
    def rho(self):
        return self.vals_dict["rho"]

    @rho.setter
    def rho(self, value):
        self.vals_dict["rho"] = value

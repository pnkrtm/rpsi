import numpy as np

from objects.models.Models import SeismicModel1D
from fmodeling.seismic.dynamic.reflection import calculate_reflection_for_depth
from objects.seismic.observation import Observation, Source, Receiver
from fmodeling.seismic.ray_tracing.case_1D.forward_tracing1D import calculate_rays


def refraction_occuracy():
    vp = np.array([1000, 2000, 2500])
    vs = np.array([500, 1000, 1250])
    rho = np.array([1800, 1850, 1900])
    h = np.array([500, 500], 'int')
    dx = 264.5

    model = SeismicModel1D(vp, vs, rho, h)
    depths = model.get_depths()

    nrec = 1
    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x * dx, 0, 0) for x in range(1, nrec+1)]
    observe = Observation(sources, receivers)
    refl_index = 2

    rays_p = calculate_rays(observe, model, 'vp')
    curve_PP2 = calculate_reflection_for_depth(depths[refl_index], model, 'vp', 'PdPu', rays_p, refl_index,
                                               use_universal_matrix=False, calculate_refraction_flag=True)

if __name__ == '__main__':
    refraction_occuracy()

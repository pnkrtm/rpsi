import numpy as np
import time

from Objects.Models.Models import SeismicModel1D
from ForwardModeling.Seismic.Dynamic.Reflection import calculate_reflection_for_depth, calculate_reflections, \
    calculate_reflections_vectorized
from Objects.Observation import Observation, Source, Receiver
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays


def reflections_p_identity_1():
    vp = np.array([1200, 1500, 1800, 2100])
    vs = np.array([600, 750, 900, 1050])
    rho = np.array([1600, 1700, 1800, 1900])
    h = np.array([500, 1000, 1500], 'int')
    dx = 100

    model = SeismicModel1D(vp, vs, rho, h)
    depths = model.get_depths()

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x * dx, 0, 0) for x in range(1, 20)]
    observe = Observation(sources, receivers)

    rays_p = calculate_rays(observe, model, 'vp')

    # индекс интересующей отражающей границы
    refl_index = 2

    curve_PP1 = calculate_reflection_for_depth(depths[refl_index], model, 'vp', 'PdPu', rays_p, refl_index,
                                               use_universal_matrix=True)
    curve_PP2 = calculate_reflection_for_depth(depths[refl_index], model, 'vp', 'PdPu', rays_p, refl_index,
                                               use_universal_matrix=False)

    ampl1 = np.array([a.real for a in curve_PP1.amplitudes])
    ampl2 = np.array([a.real for a in curve_PP2.amplitudes])

    print(ampl1 - ampl2)


def reflections_p_identity_2():
    vp = np.array([1200, 1500, 1800, 2100])
    vs = np.array([600, 750, 900, 1050])
    rho = np.array([1600, 1700, 1800, 1900])
    h = np.array([500, 1000, 1500], 'int')
    dx = 100

    model = SeismicModel1D(vp, vs, rho, h)
    depths = model.get_depths()

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x * dx, 0, 0) for x in range(1, 20)]
    observe = Observation(sources, receivers)

    rays_p = calculate_rays(observe, model, 'vp')

    time_1 = time.time()
    curves_PP1 = calculate_reflections(model, rays_p, 'PdPu')
    time_2 = time.time()
    curves_PP2 = calculate_reflections_vectorized(model, rays_p, 'PdPu')
    time_3 = time.time()

    ampl1 = np.array([a.real for curve in curves_PP1 for a in curve.amplitudes])
    ampl2 = np.array([a.real for curve in curves_PP2 for a in curve.amplitudes])

    print(f"Ordinary calculation time: {time_2 - time_1}")
    print(f"Vectorized calculation time: {time_3 - time_2}")

    print(ampl1 - ampl2)


if __name__ == '__main__':
    reflections_p_identity_2()

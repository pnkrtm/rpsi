import matplotlib.pyplot as plt
import numpy as np
import time

from Objects.Models.Models import SeismicModel1D
from Objects.Seismic.Seismogram import Seismogram
from Objects.Seismic.Observation import Observation, Source, Receiver
from Objects.Data.RDPair import OWT
from ForwardModeling.ForwardProcessing1D import create_seismogram
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays
from ForwardModeling.Seismic.Dynamic.Reflection import calculate_reflections
from ForwardModeling.Seismic.Dynamic.Refraction import calculate_refractions, calculate_refraction_vectorized
from Visualization.Seismic import visualize_model1D, visualize_time_curves, visualize_rays_model_1D, visualize_seismogram


def test_model1D():
    vp = np.array([1200, 1500, 1800, 2100])
    vs = np.array([600, 750, 900, 1050])
    rho = np.array([1600, 1700, 1800, 1900])
    h = np.array([500, 1000, 1500], 'int')
    refl_flags = [1, 1, 1]

    model = SeismicModel1D(vp, vs, rho, h, refl_flags=refl_flags)
    dz = 100
    dx = dz
    max_depth = 2000

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x*dx, 0, 0) for x in range(1, 20)]
    observe = Observation(sources, receivers)

    wavetype = OWT.PdPu

    res_seismic = dict()
    res_seismic[wavetype] = dict()

    raytracing_start_time = time.time()

    res_seismic[wavetype]["rays"] = calculate_rays(observe, model, wavetype)
    # rays_s = calculate_rays(observe, model, 'vs')

    raytracing_stop_time = time.time()

    calculate_reflections(model, res_seismic[wavetype]["rays"], wavetype)
    # calculate_refractions(model, res_seismic[wavetype]["rays"], wavetype)
    calculate_refraction_vectorized(model, res_seismic[wavetype]["rays"], wavetype)

    rays_stop_time = time.time()

    tracelen = 1000
    dt = 0.005
    res_seismic[wavetype]["seismogram"] = create_seismogram(res_seismic[wavetype]["rays"], observe, tracelen, dt)

    print('Rays calcing time = {}'.format(rays_stop_time - raytracing_start_time))
    print('Amplitudes calcing time = {}'.format(rays_stop_time - raytracing_stop_time))
    print('Raytracing evaluation time = {}'.format(raytracing_stop_time - raytracing_start_time))

    fig, axes = plt.subplots(nrows=2, ncols=1)

    rays = res_seismic[wavetype]["rays"]
    seismic = res_seismic[wavetype]["seismogram"]

    visualize_model1D(axes[1], model, observe, max_depth, dz, 'vp')
    visualize_rays_model_1D(axes[1], rays)

    visualize_seismogram(fig, axes[0], seismic, normalize=True, wiggles=False)

    plt.show()


if __name__ == '__main__':
    test_model1D()




import matplotlib.pyplot as plt
import numpy as np
import time

from objects.Models.Models import SeismicModel1D
from objects.seismic.observation import Observation, Source, Receiver
from objects.seismic.waves import OWT
from fmodeling.seismic.RayTracing.Forward1DTracing import calculate_rays
from fmodeling.seismic.dynamic.reflection import calculate_reflections
from Visualization.Seismic import visualize_model1D, visualize_time_curves, visualize_rays_model_1D


def test_model1D():
    raytracing_start_time = time.time()

    vp = np.array([1200, 1500, 1800, 2100])
    vs = np.array([600, 750, 900, 1050])
    rho = np.array([1600, 1700, 1800, 1900])
    h = np.array([500, 1000, 1500], 'int')
    refl_flags = [1, 0, 0]

    model = SeismicModel1D.from_vp_vs_rho(h, vp, vs, rho, refl_flags)
    dz = 100
    dx = dz
    max_depth = 2000

    # vp_grid, z_values = model.get_1D_regular_grid('vp', 2000, 100)

    # ncolumns = len(vp_grid)

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x*dx, 0, 0) for x in range(1, 20)]
    observe = Observation(sources, receivers)

    rays_p = calculate_rays(observe, model, OWT.PdPu)
    # rays_s = calculate_rays(observe, model, 'vs')

    rays_stop_time = time.time()

    reflections_p = calculate_reflections(model, rays_p, OWT.PdPu)
    # reflections_s = calculate_reflections(model, rays_s, 'SdSu')

    raytracing_stop_time = time.time()

    print('Rays calcing time = {}'.format(rays_stop_time- raytracing_start_time))
    print('Amplitudes calcing time = {}'.format(raytracing_stop_time - rays_stop_time))
    print('Raytracing evaluation time = {}'.format(raytracing_stop_time - raytracing_start_time))

    fig, axes = plt.subplots(nrows=2, ncols=1)

    visualize_model1D(axes[1], model, observe, max_depth, dz, 'vp')
    visualize_rays_model_1D(axes[1], rays_p)
    # axes[1, 0].title('model and rays for p-waves')

    # visualize_model1D(axes[1, 1], model, observe, max_depth, dz, 'vs')
    # visualize_rays_model_1D(axes[1, 1], rays_s)

    visualize_time_curves(axes[0], model, rays_p, observe)
    # visualize_time_curves(axes[0, 1], model, rays_s, observe)

    plt.show()

    # vp_grid = np.array([vp_grid] * ncolumns).T
    # plt.imshow(vp_grid, extent=([min(z_values), max(z_values), max(z_values), min(z_values)]), aspect='auto')
    # plt.colorbar()
    # plt.title('Model and ray paths for p-waves')
    #
    # for ray in rays:
    #     plt.plot(ray.x_points, ray.z_points)
    #
    # plt.show()
    #
    # visualize_time_curves(plt, model, rays, observe)
    # plt.show()


if __name__ == '__main__':
    test_model1D()




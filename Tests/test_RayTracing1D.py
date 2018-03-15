import matplotlib.pyplot as plt
import numpy as np

from Objects.Models.Models import SeismicModel1D
from Objects.Observation import Observation, Source, Receiver
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays
from Visualization.Seismic import visualize_model1D, visualize_time_curves, visualize_rays_model_1D


def test_model1D():
    vp = [1500, 2500, 3000]
    vs = [700, 1000, 1200]
    rho = [2700, 2700, 2800]
    h = [500, 1000]

    model = SeismicModel1D(vp, vs, rho, h)
    dz = 100
    dx = dz
    max_depth = 2000

    # vp_grid, z_values = model.get_1D_regular_grid('vp', 2000, 100)

    # ncolumns = len(vp_grid)

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x*dz, 0, 0) for x in range(1, 20)]
    observe = Observation(sources, receivers)
    rays_p = calculate_rays(observe, model, 'vp')
    rays_s = calculate_rays(observe, model, 'vs')

    fig, axes = plt.subplots(nrows=2, ncols=2)

    visualize_model1D(axes[1, 0], model, max_depth, dz, 'vp')
    visualize_rays_model_1D(axes[1, 0], rays_p)
    # axes[1, 0].title('model and rays for p-waves')

    visualize_model1D(axes[1, 1], model, max_depth, dz, 'vs')
    visualize_rays_model_1D(axes[1, 1], rays_s)
    # axes[1, 1].title('model and rays for s-waves')

    visualize_time_curves(axes[0, 0], model, rays_p, observe)
    visualize_time_curves(axes[0, 1], model, rays_s, observe)

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




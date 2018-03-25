import numpy as np


def visualize_model1D(plt, model, max_depth, dz, vel_type):
    vp_grid, z_values = model.get_1D_regular_grid(vel_type, max_depth, dz)
    ncolumns = len(vp_grid)

    vp_grid = np.array([vp_grid] * ncolumns).T
    plt.imshow(vp_grid, extent=([min(z_values), max(z_values), max(z_values), min(z_values)]), aspect='auto')


def visualize_rays_model_1D(plt, rays):
    for ray in rays:
        plt.plot(ray.x_points, ray.z_points)


def visualize_time_curves(plt, model, rays, observe):
    x = observe.get_x_geometry()

    depths = model.get_depths()

    for d in depths[1:]:
        rays_ = [r for r in rays if r.get_reflection_z() == d]
        times = [r.time for r in rays_]

        plt.plot(x, times)


def visualize_reflection_amplitudes(plt, reflections, absc='angle'):

    for r in reflections:

        if absc == 'angle':
            x = r.angles

        else:
            x = r.offsets

        y = [ampl.real for ampl in r.amplitudes]

        plt.plot(x, y)



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


def visualize_seismogram(plt, seism, normalize=False, fill_negative=False, wigles=True):

    x = seism.get_time_range()
    offsets = seism.get_offsets()

    if wigles:
        i = 0

        for offset in offsets:
            offset /= 30
            y = np.array(seism.traces[i].values)

            if normalize:
                max_y = max(y)
                y /= max_y
                # y *= 10

            y += offset
            plt.plot(y, x, 'k-')
            if fill_negative:
                plt.fill_betweenx(x, offset, y, where= y>offset, color='k')
            i += 1

        plt.set_ylim(x[-1], 0)

    else:
        values = np.array([t.values for t in seism.traces]).T

        x_vals = offsets[::-1]
        y_vals = seism.traces[0].times

        plt.imshow(values, extent=([min(x_vals), max(x_vals), max(y_vals), min(y_vals)]),
                   aspect='auto', cmap='Greys')



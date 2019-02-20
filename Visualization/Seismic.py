import numpy as np
import itertools


def visualize_model1D(plt, model, observe, max_depth, dz, vel_type, only_boundaries=False):

    if only_boundaries:
        xmin = 0
        xmax = observe.xmax

        depths = model.get_depths()

        for d in depths:
            plt.plot([xmin, xmax], [d, d], 'k', linewidth=1)

    else:
        vp_grid, z_values = model.get_1D_regular_grid(vel_type, max_depth, dz)
        ncolumns = len(vp_grid)

        vp_grid = np.array([vp_grid] * ncolumns).T
        plt.imshow(vp_grid, extent=([min(z_values), max(z_values), max(z_values), min(z_values)]), aspect='auto', cmap='jet')


def visualize_model_wellogs(plt, model, vel_type, linestyle='-', linewidth=4, legend_label='default label', scale=1):
    v = model.get_param(vel_type)
    v = np.array(list(itertools.chain(*zip(v, v)))) * scale

    depths = model.get_depths()[1:]
    depths = list(itertools.chain(*zip(depths, depths)))
    depths = np.append([0], depths)
    depths = np.append(depths, depths[-1] + 300)

    plt.plot(v, depths, linestyle, linewidth=linewidth, label=legend_label)


def visualize_rays_model_1D(plt, rays, linewidth=1):
    for ray in rays:
        plt.plot(ray.x_points, ray.z_points, 'k', linewidth=linewidth)


def visualize_time_curves(plt, model, rays, observe, depth_index=None, linewidth=4, linestyle='-'):
    x = observe.get_x_geometry()

    depths = model.get_depths()
    i = 1

    for d in depths[1:]:
        if depth_index is not None and (i-1) != depth_index:
            i += 1
            continue

        rays_ = [r for r in rays if r.reflection_z == d]
        times = [r.time for r in rays_]

        plt.plot(x, times, label='Граница {}'.format(i), linewidth=linewidth, linestyle=linestyle)
        i += 1


def visualize_reflection_amplitudes(plt, reflections, reflection_index=None, absc='angle', linewidth=4, linestyle='-'):
    i = 1
    for r in reflections:

        if reflection_index is not None and (i - 1) != reflection_index:
            i += 1
            continue

        if absc == 'angle':
            x = r.angles

        else:
            x = r.offsets

        y = [ampl.real for ampl in r.amplitudes]

        plt.plot(x, y, label='Граница {}'.format(i), linewidth=linewidth, linestyle=linestyle)
        i += 1


def visualize_seismogram(plt, seism, normalize=False, fill_negative=False, wigles=True, gain=1):

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
            y *= gain

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



import cmath
import numpy as np

from Objects.Rays import Ray1D


def eq_to_solve(p, v, h, x_2):
    number_of_layers = len(v)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

            if i > number_of_layers - 2:
                jj = 2 * number_of_layers - 3 - i
            else:
                jj = i

            xx = xx + (p * v[jj] * h[jj]) / cmath.sqrt(1 - p * p * v[jj] * v[jj])

    res = xx - x_2

    return res


def deriv_eq_to_solve(p, v, h):
    number_of_layers = len(v)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 2:
            j = 2 * number_of_layers - 3 - i
        else:
            j = i

        a1 = (v[j] * h[j])
        a2 = cmath.sqrt(pow((1 - p * p * v[j] * v[j]), 3))
        xx = xx + a1 / a2


    res = xx

    return res


def solve_for_p(p_start, v, h, x_2, tol=0.1):
    pn = p_start

    while True:
        if pn.imag != 0:
            alpha = cmath.asin(pn * v[0])
            alpha = alpha / 100
            pn = cmath.sin(alpha) / v[0]

        pn1 = pn - eq_to_solve(pn, v, h, x_2) / deriv_eq_to_solve(pn, v, h)
        res = abs(eq_to_solve(pn1, v, h, x_2))
        if res < tol:
            break
        else:
            pn = pn1

    p = pn1

    return p.real


def forward_rtrc(v, h, p):
    number_of_layers = len(v)

    x = np.zeros(2 * number_of_layers - 1)
    z = np.zeros(2 * number_of_layers - 1)
    t = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 2:
            jj = 2 * number_of_layers - 3 - i
            dz = -h[jj]
        else:
            jj = i
            dz = h[jj]

        dx = (p * v[jj] * h[jj]) / (np.sqrt(1 - p * p * v[jj] * v[jj]))
        x[i + 1] = x[i] + dx
        z[i + 1] = z[i] + dz
        t = t + np.sqrt(dx * dx + dz * dz) / v[jj]

    return x, z, t


def calculate_rays_for_layer(model, observ, velocity_type, layer_index):
    rays = []
    p = np.sin(np.pi / 4) / model.get_param(velocity_type, index_start=0, index_finish=1)[0]

    for receiver in observ.receivers:
        p_start = p
        p = solve_for_p(p_start, model.get_param(velocity_type, index_finish=layer_index + 1), model.get_param('h', index_finish=layer_index),
                        receiver.x)

        x, z, t = forward_rtrc(model.get_param(velocity_type, index_finish=layer_index + 1), model.get_param('h', index_finish=layer_index),
                               p)
        rays.append(Ray1D(x, z, t, p))

    return rays


def calculate_rays_for_layer_mp_helper(args):
    return calculate_rays_for_layer(*args)


def calculate_rays(observ, model, velocity_type='vp'):
    rays = []

    for i in range(1, model.get_number_of_layers()):
        rays = np.append(rays, calculate_rays_for_layer(model, observ, velocity_type, i))
        # for source in observ.sources:
        #     for receiver in observ.receivers:
        #         p_start = p
        #
        #         p = solve_for_p(p_start, model.get_param(velocity_type, index_finish=i+1), model.get_param('h', index_finish=i), receiver.x)
        #
        #         x, z, t = forward_rtrc(model.get_param(velocity_type, index_finish=i+1), model.get_param('h', index_finish=i), p)
        #         rays.append(Ray1D(x, z, t, p))

    return rays



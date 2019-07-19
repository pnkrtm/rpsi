import cmath
import numpy as np
from collections import OrderedDict

from Objects.Seismic.Rays import Ray1D
from Objects.Data.RDPair import OWT, get_down_up_vel_types


def eq_to_solve(p, vd, vu, h, x_2):
    assert (len(vd) == len(vu))
    number_of_layers = len(vd)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

            if i > number_of_layers - 2:
                jj = 2 * number_of_layers - 3 - i
                v = vu
            else:
                jj = i
                v = vd

            xx = xx + (p * v[jj] * h[jj]) / cmath.sqrt(1 - p * p * v[jj] * v[jj])

    res = xx - x_2

    return res


def deriv_eq_to_solve(p, vd, vu, h):
    assert (len(vd) == len(vu))
    number_of_layers = len(vd)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 2:
            j = 2 * number_of_layers - 3 - i
            v = vu

        else:
            j = i
            v = vd

        a1 = (v[j] * h[j])
        a2 = cmath.sqrt(pow((1 - p * p * v[j] * v[j]), 3))
        xx = xx + a1 / a2

    res = xx

    return res


def solve_for_p(p_start, vd, vu, h, x_2, tol=0.001):
    pn = p_start

    while True:
        if pn.imag != 0:
            alpha = cmath.asin(pn * vd[0])
            alpha = alpha / 100
            pn = cmath.sin(alpha) / vd[0]

        pn1 = pn - eq_to_solve(pn, vd, vu, h, x_2) / deriv_eq_to_solve(pn, vd, vu, h)
        res = abs(eq_to_solve(pn1, vd, vu, h, x_2))
        if res < tol:
            break
        else:
            pn = pn1

    p = pn1

    return p.real


def forward_rtrc(vd, vu, h, p):
    assert(len(vd) == len(vu))
    number_of_layers = len(vd)

    x = np.zeros(2 * number_of_layers - 1)
    z = np.zeros(2 * number_of_layers - 1)
    t = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 2:
            jj = 2 * number_of_layers - 3 - i
            dz = -h[jj]
            v = vu
        else:
            jj = i
            dz = h[jj]
            v = vd

        dx = (p * v[jj] * h[jj]) / (np.sqrt(1 - p * p * v[jj] * v[jj]))
        x[i + 1] = x[i] + dx
        z[i + 1] = z[i] + dz
        t = t + np.sqrt(dx * dx + dz * dz) / v[jj]

    return x, z, t


def calculate_rays_for_layer(model, observ, owt, layer_index):
    rays = []
    vel_types = get_down_up_vel_types(owt)
    p = np.sin(np.pi / 4) / model.get_single_param(vel_types['down'], index_start=0, index_finish=1)[0]

    for receiver in observ.receivers:
        p_start = p
        p = solve_for_p(p_start,
                        model.get_single_param(vel_types['down'], index_finish=layer_index + 1),
                        model.get_single_param(vel_types['up'], index_finish=layer_index + 1),
                        model.get_single_param('h', index_finish=layer_index),
                        receiver.x)

        x, z, t = forward_rtrc(model.get_single_param(vel_types['down'], index_finish=layer_index + 1),
                               model.get_single_param(vel_types['up'], index_finish=layer_index + 1),
                               model.get_single_param('h', index_finish=layer_index),
                               p)
        rays.append(Ray1D(owt, x, z, t, p, receiver.x))

    return rays


def calculate_rays_for_layer_mp_helper(args):
    return calculate_rays_for_layer(*args)


def calculate_rays(observ, model, owt):
    rays = OrderedDict()

    # TODO check multiwaves condition
    for i, refl in zip(range(1, model.get_number_of_layers()), model.refl_flags):

        if refl:
            rays[i] = calculate_rays_for_layer(model, observ, owt, i)

    return rays

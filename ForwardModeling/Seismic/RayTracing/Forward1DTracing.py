import cmath
import numpy as np

def eq_to_solve(p, v, h, x_2):
    number_of_layers = len(v)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

            if i > number_of_layers - 1:
                jj = 2 * number_of_layers - 1 - i
            else:
                jj = i

            xx = xx + (p * v(jj) * h(jj)) / cmath.sqrt(1 - p * p * v[jj] * v[jj])

    res = xx - x_2

    return res


def deriv_eq_to_solve(p, v, h):
    number_of_layers = len(v)
    xx = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 1:
            j = 2 * number_of_layers - 1 - i
        else:
            j = i

        xx = xx + (v[j] * h[j]) / cmath.sqrt((1 - p * p * v[j] * v[j]) ** 3)

    res = xx

    return res


def solve_for_p(p_start, v, h, x_2, tol):
    pn = p_start
    pn1 = 0
    while True:
        if pn.imag != 0:
            alpha = cmath.asin(pn * v[1])
            alpha = alpha / 2
            pn = cmath.sin(alpha) / v[1]

        pn1 = pn - eq_to_solve(pn, v, h, x_2) / deriv_eq_to_solve(pn, v, h)
        res = abs(eq_to_solve(pn1, v, h, x_2))
        if res < tol:
            break
        else:
            pn = pn1

    p = pn1

    return p


def forward_rtrc(v, h, p):
    number_of_layers = len(v)

    x = np.zeros(2 * number_of_layers - 1)
    z = np.zeros(2 * number_of_layers - 1)
    t = 0

    for i in range(2 * (number_of_layers - 1)):

        if i > number_of_layers - 1:
            jj = 2 * number_of_layers - 1 - i
            dz = -h(jj)
        else:
            jj = i
            dz = h[jj]

        dx = (p * v[jj] * h[jj]) / (np.sqrt(1 - p * p * v[jj] * v[jj]))
        x[i + 1] = x[i] + dx
        z[i + 1] = z[i] + dz
        t = t + np.sqrt(dx * dx + dz * dz) / v[jj]

    return x, z, t
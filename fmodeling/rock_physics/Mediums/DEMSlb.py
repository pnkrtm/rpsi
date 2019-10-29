import numpy as np
from numba import jit

from fmodeling.Utils.RungeKutta45 import rkdumb


def CalculateCoeffs(k, mu, k_incl, mu_incl, asp_curr):
    theta = 0
    fn = 0
    nu = 0
    r = 0
    a = 0
    b = 0
    asp2 = 0
    f1 = f2 = f3 = f4 = f5 = f6 = f7 = f8 = f9 = 0

    asp2 = asp_curr * asp_curr

    # Calculating coefficients theta and fn
    if asp_curr < 1:
        theta = (asp_curr / (pow((1 - asp2), 1.5)))*(np.arccos(asp_curr) - asp_curr * np.sqrt(1 - asp2))
        fn = ((asp_curr * asp_curr) / (1 - asp2))*(3 * theta - 2)

    elif asp_curr > 1:
        theta = (asp_curr / (pow((asp2 - 1), 1.5)))*(asp_curr * np.sqrt(asp2 - 1) - np.arccosh(asp_curr))
        fn = ((asp2) / (asp2 - 1))*(2 - 3 * theta)

    else:
        raise ValueError("Aspect ration is equal to 1!")
    # Calculating coefficients f1, f2, f3, f4, f5, f6, f7, f8, f9

    nu = (3 * k - 2 * mu) / (2 * (3 * k + mu))
    r = (1 - 2 * nu) / (2 * (1 - nu))

    a = mu_incl / mu - 1
    b = (k_incl / k - mu_incl / mu) / 3

    f1 = 1 + a * (1.5 * (fn + theta) - r * (1.5 * fn + 2.5 * theta - (4. / 3.)))
    f2 = 1 + a * (1 + 1.5 * (fn + theta) - r * (3 * fn + 5 * theta) / 2) + b * (3 - 4 * r) + \
        a * (a + 3 * b) * (3 - 4 * r) * (fn + theta - r * (fn - theta + 2 * theta * theta)) / 2
    f3 = 1 + 0.5 * a * (r * (2 - theta) + ((1 + asp2) * fn * (r - 1)) / asp2)
    f4 = 1 + 0.25 * a * (3 * theta + fn - r * (fn - theta))
    f5 = a * (r * (fn + theta - (4. / 3.)) - fn) + b * theta * (3 - 4 * r)
    f6 = 1 + a * (1 + fn - r * (fn + theta)) + b * (1 - theta) * (3 - 4 * r)
    f7 = 2 + 0.25 * a * (9 * theta + 3 * fn - r * (5 * theta + 3 * fn)) + b * theta * (3 - 4 * r)
    f8 = a * (1 - 2 * r + 0.5 * fn * (r - 1) + 0.5 * theta * (5 * r - 3)) + b * (1 - theta) * (3 - 4 * r)
    f9 = a * (fn * (r - 1) - r * theta) + b * theta * (3 - 4 * r)

    # Calculating pre- coefficients in p and q
    p = 3 * f1 / f2
    q = (2 / f3) + (1 / f4) + ((f4*f5 + f6*f7 - f8*f9) / (f2*f4))
    # Calculating coefficients p and q
    p = p / 3
    q = q / 5

    return p, q


def Equation(t, y, eq_params):

    k, mu = y
    k_incl_curr, mu_incl_curr, asp_curr = eq_params

    yout = np.array([0, 0])

    p, q = CalculateCoeffs(k, mu, k_incl_curr, mu_incl_curr, asp_curr)

    yout[0] = ((k_incl_curr - k)*p) / (1 - t)
    yout[1] = ((mu_incl_curr - mu)*q) / (1 - t)

    return yout


def DEM(k0, mu0, k_incl, mu_incl, por, asp, phic=1):
    j = 0

    tfinal = 0
    t0 = 0
    h = 0
    gl = 0

    correctPorosity = 1

    k_curr = k0
    mu_curr = mu0

    asp_curr = asp
    if asp_curr == 1:
        asp_curr = 0.99

    k_incl_curr = k_incl
    mu_incl_curr = mu_incl

    t0 = 0

    por_curr = por

    if (por < phic):
        tfinal = por / phic

        if (phic < 1):
            fgl = mu_incl_curr * (9. * k_incl_curr + 8. * mu_incl_curr) / \
                (6. * (k_incl_curr + 2. * mu_incl_curr))
            k_incl_curr = k_incl_curr + (1. - phic) * (k_curr - k_incl_curr) * \
                (k_incl_curr + 4. * mu_incl_curr / 3.) / (k_incl_curr + 4. * (mu_incl_curr / 3.) +
                    phic * (k_curr - k_incl_curr))
            mu_incl_curr = mu_incl_curr + (mu_curr - mu_incl_curr) * (1. - phic) * \
                (mu_incl_curr + fgl) / (mu_incl_curr + fgl + phic * (mu_curr - mu_incl_curr))

        h = (tfinal - t0) / (6)

        y = [k_curr, mu_curr]

        yout = Equation(t0, y, [k_incl_curr, mu_incl_curr, asp_curr])

        k_calced, mu_calced = rkdumb(y, t0, tfinal, 6, Equation, [k_incl_curr, mu_incl_curr, asp_curr])

    else:
        fgl = mu_incl_curr * (9. * k_incl_curr + 8. * mu_incl_curr) / \
            (6. * (k_incl_curr + 2. * mu_incl_curr))
        k_calced = k_incl_curr + (1. - por_curr) * (k_curr - k_incl_curr) * \
            (k_incl_curr + 4. * mu_incl_curr / 3.) / (k_incl_curr + 4. * (mu_incl_curr / 3.) +
                por_curr * (k_curr - k_incl_curr))
        mu_calced = mu_incl_curr + (mu_curr - mu_incl_curr) * (1. - por_curr) * \
            (mu_incl_curr + fgl) / (mu_incl_curr + fgl + por_curr * (mu_curr - mu_incl_curr))

    return k_calced, mu_calced


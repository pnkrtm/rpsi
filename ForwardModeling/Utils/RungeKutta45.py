import numpy as np

from numba import jit

a2 = 0.2
a3 = 0.3
a4 = 0.6
a5 = 1.0
a6 = 0.875

b21 = 0.2
b31 = 3.0 / 40.0
b32 = 9.0 / 40.0
b41 = 0.3
b42 = -0.9
b43 = 1.2
b51 = -11.0 / 54.0
b52 = 2.5
b53 = -70.0 / 27.0
b54 = 35.0 / 27.0
b61 = 1631.0 / 55296.0
b62 = 175.0 / 512.0
b63 = 575.0 / 13824.0
b64 = 44275.0 / 110592.0
b65 = 253.0 / 4096.0
c1 = 37.0 / 378.0
c3 = 250.0 / 621.0
c4 = 125.0 / 594.0
c6 = 512.0 / 1771.0
dc5 = -277.00 / 14336.0

dc1 = c1 - 2825.0 / 27648.0
dc3 = c3 - 18575.0 / 48384.0
dc4 = c4 - 13525.0 / 55296.0
dc6 = c6 - 0.25


def rk4(y, dydx, x, h, Equation, eq_param):
    hh = h*0.5
    h6 = h / 6.0
    xh = x + hh
    yt = y + hh*dydx

    dyt = Equation(xh, yt, eq_param)

    yt = y + hh*dyt
    dym = Equation(xh, yt, eq_param)

    yt = y + h*dym
    dym += dyt

    dyt = Equation(x + h, yt, eq_param)

    yout = y + h6*(dydx + dyt + 2.0*dym)

    return yout


def rkdumb(y_start, t0, tfinal, nstep, Equation, eq_param):
    v = y_start

    x = t0
    h = (tfinal - t0) / nstep

    for k in range(nstep):
        dv = Equation(x, v, eq_param)
        vout = rk4(v, dv, x, h, Equation, eq_param)
        if h == 0:
            v = vout
            yout = vout

            return yout

        if float(x + h) == x:
            raise ValueError("RungeKutta error")

        x += h

        v = vout
        yout = vout

    return yout

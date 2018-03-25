import numpy as np


def vp_from_KGRho(K, G, rho):
    return np.sqrt((K + 4.0*G/3.0)/rho)


def vs_from_GRho(G, rho):
    return np.sqrt(G/rho)


def G_from_KPoissonRatio(K, nu):
    return K / (2*(nu + 1))


def G_from_VsDensity(vs, rho):
    return vs * vs * rho


def K_from_VpVsDensity(vp, vs, rho):
    return rho * (vp*vp - (4/3) * vs * vs)
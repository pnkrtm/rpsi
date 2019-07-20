import numpy as np

from ForwardModeling.RockPhysics.Models import xu_payne_model


def simple_calculate_test():
    Km = np.array([65])
    Gm = np.array([28])
    rho_m = np.array([2.71])

    Ks = 46
    Gs = 23
    rho_s = 2.43

    Kf = 2.41
    rho_f = 0.950

    phi = 0.1
    phi_s = 0.1

    vp, vs, rho = xu_payne_model(Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m)

    print(vp, vs, rho)


if __name__ == '__main__':
    simple_calculate_test()

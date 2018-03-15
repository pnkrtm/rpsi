import numpy as np
from ForwardModeling.ForwardProcessing1D import forward
from Inversion.Inversion1D import inverse
from Tests.test_ForwardProcessing1D import get_model_1

def main():
    Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    nlayers = 8
    dx = 100
    nx = 20
    x_rec = [i * dx for i in range(1, nx)]

    observe, model, rays_observed_p, rays_observed_s = forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, False)

    # data_start =

    inverse(nlayers, Km, Gm, Ks, Gs, Kf, h, x_rec, rays_observed_p, rays_observed_s, )


if __name__ == '__main__':
    main()
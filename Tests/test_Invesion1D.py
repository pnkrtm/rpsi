import sys
sys.path.append('../')

import numpy as np
from ForwardModeling.ForwardProcessing1D import forward
from Inversion.Strategies.Inversion1D import inverse
from Tests.test_ForwardProcessing1D import get_model_1

import time

def main():
    Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    nlayers = 8
    dx = 100
    nx = 20
    x_rec = [i * dx for i in range(1, nx)]


    print('Calculating DEM modeling...')
    observe, model, rays_observed_p, rays_observed_s = \
        forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f,
                                                               rho_m, h, x_rec, display_stat=True,
            visualize_res=False, calc_reflection_p=True, calc_reflection_s=True)
    print('Forward calculated!')

    true_model = []
    true_model = np.append(true_model, Kf)
    true_model = np.append(true_model, phi)
    true_model = np.append(true_model, phi_s)
    true_model = np.append(true_model, rho_s)
    true_model = np.append(true_model, rho_f)
    true_model = np.append(true_model, rho_m)

    data_start_ = \
    [
        [(0, 3)]*nlayers,  # Kf
        [(0, 0.2)]*nlayers,  # phi
        [(0, 0.2)]*nlayers,  # phi_s
        [(2.2, 2.6)]*nlayers,  # rho_s
        [(0, 3)]*nlayers,  # rho_f
        [(2, 3)]*nlayers,  # rho_m
    ]

    data_start = [item for sublist in data_start_ for item in sublist]

    # for d in data_start_:
    #     data_start = np.append(data_start, d)

    inversion_start_time = time.time()
    print('Calculating inversion...')
    inversed_model = inverse(nlayers, Km, Gm, Ks, Gs, h, x_rec,
                             rays_observed_p, rays_observed_s,
                             reflection_observed_p, reflection_observed_s,
                             data_start,
                             opt_type='de',
            use_rays_p=True, use_rays_s=True,
    use_reflection_p = True, use_reflection_s = True)

    print('Inversion calculated!')
    inversion_end_time = time.time()

    print('Inversion duration in minutes: {}'.format((inversion_end_time - inversion_start_time)/60))

    print(inversed_model)
    print(np.column_stack((true_model, inversed_model)))


if __name__ == '__main__':
    main()
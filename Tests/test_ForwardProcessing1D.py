import numpy as np

from ForwardModeling.ForwardProcessing1D import forward, forward_with_trace_calcing
from ForwardModeling.RockPhysics.Tools import G_from_KPoissonRatio, G_from_VsDensity, K_from_VpVsDensity

def get_model_1():
    '''
    Модель без составных минералов в слоях
    информация об упругих модулях из https://cf.ppt-online.org/files/slide/b/brl4LwEI1VGRNvdgyfAWCe7UBkHToh6m5DcZK3/slide-7.jpg
    информация о скоростях из http://geum.ru/next/images/210276-nomer-m4971715b.png
    информация о плотностях из http://www.astronet.ru/db/msg/1173309/page4.html
    :return:
    '''
    # первый слой - песчаник, мощность 1000 м
    Km_1 = np.array([K_from_VpVsDensity(2, 1, 2.71)])
    Gm_1 = np.array([G_from_VsDensity(1, 2.71)])
    rho_m_1 = np.array([2.71])

    Ks_1 = 0
    Gs_1 = 0
    rho_s_1 = 0

    Kf_1 = 0
    rho_f_1 = 0

    phi_1 = 0
    phi_s_1 = 0

    h_1 = 1000

    # второй слой - песчаник+глина, мощность 300 м
    Km_2 = np.array([K_from_VpVsDensity(3, 1.5, 2.75)])
    Gm_2 = np.array([G_from_VsDensity(1.5, 2.75)])
    rho_m_2 = np.array([2.75])

    Ks_2 = np.array([K_from_VpVsDensity(2, 0.4, 2.43)])
    Gs_2 = np.array([G_from_VsDensity(0.4, 2.43)])
    rho_s_2 = 2.43

    Kf_2 = 0
    rho_f_2 = 0

    phi_2 = 0
    phi_s_2 = 0.05

    h_2 = 300

    # третий слой - известняк, мощность 200 м
    Km_3 = np.array([K_from_VpVsDensity(3.5, 2, 2.8)])
    Gm_3 = np.array([G_from_VsDensity(2, 2.8)])
    rho_m_3 = np.array([2.8])

    Ks_3 = 0
    Gs_3 = 0
    rho_s_3 = 0

    Kf_3 = 0
    rho_f_3 = 0

    phi_3 = 0
    phi_s_3 = 0

    h_3 = 200

    # четвертый слой - известняк+глина+газ, мощность 100 м
    Km_4 = np.array([K_from_VpVsDensity(4, 2.5, 2.8)])
    Gm_4 = np.array([G_from_VsDensity(2.5, 2.8)])
    rho_m_4 = np.array([2.8])

    Ks_4 = np.array([K_from_VpVsDensity(2, 0.4, 2.43)])
    Gs_4 = np.array([G_from_VsDensity(0.4, 2.43)])
    rho_s_4 = 2.43

    Kf_4 = 0
    rho_f_4 = 0

    phi_4 = 0.15
    phi_s_4 = 0.05

    h_4 = 100

    # пятый слой - известняк+глина+нефть, мощность 100 м
    Km_5 = np.array([K_from_VpVsDensity(4, 2.5, 2.8)])
    Gm_5 = np.array([G_from_VsDensity(2.5, 2.8)])
    rho_m_5 = np.array([2.8])

    Ks_5 = np.array([K_from_VpVsDensity(2, 0.4, 2.43)])
    Gs_5 = np.array([G_from_VsDensity(0.4, 2.43)])
    rho_s_5 = 2.43

    Kf_5 = 2.41
    rho_f_5 = 0.95

    phi_5 = 0.1
    phi_s_5 = 0.05

    h_5 = 100

    # шестой слой - известняк+глина+вода, мощность 100 м
    Km_6 = np.array([K_from_VpVsDensity(4, 2.5, 2.8)])
    Gm_6 = np.array([G_from_VsDensity(2.5, 2.8)])
    rho_m_6 = np.array([2.8])

    Ks_6 = np.array([K_from_VpVsDensity(2, 0.4, 2.43)])
    Gs_6 = np.array([G_from_VsDensity(0.4, 2.43)])
    rho_s_6 = 2.43

    Kf_6 = 2
    rho_f_6 = 1

    phi_6 = 0.07
    phi_s_6 = 0.05

    h_6 = 100

    # седьмой слой - глина + известняк, мощность 50 м
    Km_7 = np.array([K_from_VpVsDensity(4, 2.5, 2.8)])
    Gm_7 = np.array([G_from_VsDensity(2.5, 2.8)])
    rho_m_7 = np.array([2.8])

    Ks_7 = np.array([K_from_VpVsDensity(2, 0.4, 2.43)])
    Gs_7 = np.array([G_from_VsDensity(0.4, 2.43)])
    rho_s_7 = 2.43

    Kf_7 = 0
    rho_f_7 = 0

    phi_7 = 0
    phi_s_7 = 0.3

    h_7 = 50

    # последний слой - известняк
    Km_8 = np.array([K_from_VpVsDensity(4.5, 2.7, 2.85)])
    Gm_8 = np.array([G_from_VsDensity(2.7, 2.85)])
    rho_m_8 = np.array([2.85])

    Ks_8 = 0
    Gs_8 = 0
    rho_s_8 = 0

    Kf_8 = 0
    rho_f_8 = 0

    phi_8 = 0
    phi_s_8 = 0

    Km = np.array([Km_1, Km_2, Km_3, Km_4, Km_5, Km_6, Km_7, Km_8])
    Gm = np.array([Gm_1, Gm_2, Gm_3, Gm_4, Gm_5, Gm_6, Gm_7, Gm_8])
    rho_m = np.array([rho_m_1, rho_m_2, rho_m_3, rho_m_4, rho_m_5, rho_m_6, rho_m_7, rho_m_8])

    Ks = np.array([Ks_1, Ks_2, Ks_3, Ks_4, Ks_5, Ks_6, Ks_7, Ks_8])
    Gs = np.array([Gs_1, Gs_2, Gs_3, Gs_4, Gs_5, Gs_6, Gs_7, Gs_8])
    rho_s = np.array([rho_s_1, rho_s_2, rho_s_3, rho_s_4, rho_s_5, rho_s_6, rho_s_7, rho_s_8])

    Kf = np.array([Kf_1, Kf_2, Kf_3, Kf_4, Kf_5, Kf_6, Kf_7, Kf_8])
    rho_f = np.array([rho_f_1, rho_f_2, rho_f_3, rho_f_4, rho_f_5, rho_f_6, rho_f_7, rho_f_8])

    phi = np.array([phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8])
    phi_s = np.array([phi_s_1, phi_s_2, phi_s_3, phi_s_4, phi_s_5, phi_s_6, phi_s_7, phi_s_8])

    h = np.array([h_1, h_2, h_3, h_4, h_5, h_6, h_7])

    return Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h


def main():
    Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    nlayers = 8
    dx = 50
    nx = 20
    x_rec = [i*dx for i in range(1, nx)]

    forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
            display_stat=True, visualize_res=True,
            calc_rays_p=True, calc_rays_s=True,
            calc_reflection_p=True, calc_reflection_s=False
            )

    # forward_with_trace_calcing(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
    #                            dt=3e-03, trace_len=1500,
    #         display_stat=True, visualize_res=False,
    #         use_p_waves=True, use_s_waves=True,
    #                            visualize_seismograms=True
    #         )




if __name__ == '__main__':
    main()

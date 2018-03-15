import numpy as np

from ForwardModeling.ForwardProcessing1D import forward
from ForwardModeling.RockPhysics.Tools import G_from_KPoissonRatio

def get_model_1():
    '''
    Модель без составных минералов в слоях
    информация об упругих модулях из https://cf.ppt-online.org/files/slide/b/brl4LwEI1VGRNvdgyfAWCe7UBkHToh6m5DcZK3/slide-7.jpg
    информация о плотностях из http://www.astronet.ru/db/msg/1173309/page4.html
    :return:
    '''
    # первый слой - известняк, мощность 1000 м
    Km_1 = np.array([50])
    Gm_1 = np.array([G_from_KPoissonRatio(50, 0.2)])
    rho_m_1 = np.array([2.71])

    Ks_1 = 0
    Gs_1 = 0
    rho_s_1 = 0

    Kf_1 = 0
    rho_f_1 = 0

    phi_1 = 0
    phi_s_1 = 0

    h_1 = 1000

    # второй слой - известняк+глина, мощность 300 м
    Km_2 = np.array([55])
    Gm_2 = np.array([G_from_KPoissonRatio(55, 0.25)])
    rho_m_2 = np.array([2.8])

    Ks_2 = 46
    Gs_2 = 23
    rho_s_2 = 2.43

    Kf_2 = 0
    rho_f_2 = 0

    phi_2 = 0
    phi_s_2 = 0.15

    h_2 = 300

    # третий слой - песчаник, мощность 200 м
    Km_3 = np.array([60])
    Gm_3 = np.array([G_from_KPoissonRatio(60, 0.2)])
    rho_m_3 = np.array([2.8])

    Ks_3 = 0
    Gs_3 = 0
    rho_s_3 = 0

    Kf_3 = 0
    rho_f_3 = 0

    phi_3 = 0
    phi_s_3 = 0

    h_3 = 200

    # четвертый слой - песчаник+глина+газ, мощность 100 м
    Km_4 = np.array([60])
    Gm_4 = np.array([G_from_KPoissonRatio(60, 0.2)])
    rho_m_4 = np.array([2.8])

    Ks_4 = 46
    Gs_4 = 23
    rho_s_4 = 2.43

    Kf_4 = 0
    rho_f_4 = 0

    phi_4 = 0.15
    phi_s_4 = 0.15

    h_4 = 100

    # пятый слой - песчаник+глина+нефть, мощность 100 м
    Km_5 = np.array([60])
    Gm_5 = np.array([G_from_KPoissonRatio(60, 0.2)])
    rho_m_5 = np.array([2.8])

    Ks_5 = 46
    Gs_5 = 23
    rho_s_5 = 2.43

    Kf_5 = 2.41
    rho_f_5 = 0.95

    phi_5 = 0.15
    phi_s_5 = 0.1

    h_5 = 100

    # шестой слой - песчаник+глина+вода, мощность 100 м
    Km_6 = np.array([70])
    Gm_6 = np.array([G_from_KPoissonRatio(70, 0.15)])
    rho_m_6 = np.array([2.85])

    Ks_6 = 49
    Gs_6 = 26
    rho_s_6 = 2.47

    Kf_6 = 2
    rho_f_6 = 1

    phi_6 = 0.1
    phi_s_6 = 0.1

    h_6 = 100

    # седьмой слой - глина, мощность 50 м
    Km_7 = np.array([70])
    Gm_7 = np.array([G_from_KPoissonRatio(70, 0.15)])
    rho_m_7 = np.array([2.85])

    Ks_7 = 49
    Gs_7 = 26
    rho_s_7 = 2.47

    Kf_7 = 0
    rho_f_7 = 0

    phi_7 = 0
    phi_s_7 = 0.95

    h_7 = 50

    # последний слой - известняк
    Km_8 = np.array([65])
    Gm_8 = np.array([G_from_KPoissonRatio(65, 0.17)])
    rho_m_8 = np.array([2.79])

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
    dx = 100
    nx = 20
    x_rec = [i*dx for i in range(1, nx)]

    forward(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec, True)




if __name__ == '__main__':
    main()

import time
import numpy as np

from fmodelling.forward_proc_1D import forward, forward_with_trace_calcing
from fmodelling.rock_physics.Tools import G_from_KPoissonRatio, G_from_VsDensity, K_from_VpVsDensity
from objects.seismic.waves import OWT
from objects.models.Models import SeismicModel1D
from objects.attributes.rock_physics.RockPhysicsAttribute import RockPhysicsAttribute
from objects.attributes.seismic.SeismicAttribute import SeismicAttribute
from objects.models.Layer1D import Layer1D, LayerOPT
from collections import OrderedDict


def get_model_1():
    '''
    Модель без составных минералов в слоях
    информация об упругих модулях из https://cf.ppt-online.org/files/slide/b/brl4LwEI1VGRNvdgyfAWCe7UBkHToh6m5DcZK3/slide-7.jpg
    информация о скоростях из http://geum.ru/next/images/210276-nomer-m4971715b.png
    информация о плотностях из http://www.astronet.ru/db/msg/1173309/page4.html
    :return:
    '''
    # первый слой - песчаник, мощность 1000 м
    Km_1 = K_from_VpVsDensity(2, 1, 2.71)
    Gm_1 = G_from_VsDensity(1, 2.71)
    rho_m_1 = 2.71

    Ks_1 = 0
    Gs_1 = 0
    rho_s_1 = 0

    Kf_1 = 0
    rho_f_1 = 0

    phi_1 = 0
    phi_s_1 = 0

    h_1 = 1000

    # второй слой - песчаник+глина, мощность 300 м
    Km_2 = K_from_VpVsDensity(3, 1.5, 2.75)
    Gm_2 = G_from_VsDensity(1.5, 2.75)
    rho_m_2 = 2.75

    Ks_2 = K_from_VpVsDensity(2, 0.4, 2.43)
    Gs_2 = G_from_VsDensity(0.4, 2.43)
    rho_s_2 = 2.43

    Kf_2 = 0
    rho_f_2 = 0

    phi_2 = 0
    phi_s_2 = 0.05

    h_2 = 300

    # третий слой - известняк, мощность 200 м
    Km_3 = K_from_VpVsDensity(3.5, 2, 2.8)
    Gm_3 = G_from_VsDensity(2, 2.8)
    rho_m_3 = 2.8

    Ks_3 = 0
    Gs_3 = 0
    rho_s_3 = 0

    Kf_3 = 0
    rho_f_3 = 0

    phi_3 = 0
    phi_s_3 = 0

    h_3 = 200

    # четвертый слой - известняк+глина+газ, мощность 100 м
    Km_4 = K_from_VpVsDensity(4, 2.5, 2.8)
    Gm_4 = G_from_VsDensity(2.5, 2.8)
    rho_m_4 = 2.8

    Ks_4 = K_from_VpVsDensity(2, 0.4, 2.43)
    Gs_4 = G_from_VsDensity(0.4, 2.43)
    rho_s_4 = 2.43

    Kf_4 = 0
    rho_f_4 = 0

    phi_4 = 0.15
    phi_s_4 = 0.05

    h_4 = 100

    # пятый слой - известняк+глина+нефть, мощность 100 м
    Km_5 = K_from_VpVsDensity(4, 2.5, 2.8)
    Gm_5 = G_from_VsDensity(2.5, 2.8)
    rho_m_5 = 2.8

    Ks_5 = K_from_VpVsDensity(2, 0.4, 2.43)
    Gs_5 = G_from_VsDensity(0.4, 2.43)
    rho_s_5 = 2.43

    Kf_5 = 2.41
    rho_f_5 = 0.95

    phi_5 = 0.1
    phi_s_5 = 0.05

    h_5 = 100

    # шестой слой - известняк+глина+вода, мощность 100 м
    Km_6 = K_from_VpVsDensity(4, 2.5, 2.8)
    Gm_6 = G_from_VsDensity(2.5, 2.8)
    rho_m_6 = 2.8

    Ks_6 = K_from_VpVsDensity(2, 0.4, 2.43)
    Gs_6 = G_from_VsDensity(0.4, 2.43)
    rho_s_6 = 2.43

    Kf_6 = 2
    rho_f_6 = 1

    phi_6 = 0.07
    phi_s_6 = 0.05

    h_6 = 100

    # седьмой слой - глина + известняк, мощность 50 м
    Km_7 = K_from_VpVsDensity(4, 2.5, 2.8)
    Gm_7 = G_from_VsDensity(2.5, 2.8)
    rho_m_7 = 2.8

    Ks_7 = K_from_VpVsDensity(2, 0.4, 2.43)
    Gs_7 = G_from_VsDensity(0.4, 2.43)
    rho_s_7 = 2.43

    Kf_7 = 0
    rho_f_7 = 0

    phi_7 = 0
    phi_s_7 = 0.3

    h_7 = 50

    # последний слой - известняк
    Km_8 = K_from_VpVsDensity(4.5, 2.7, 2.85)
    Gm_8 = G_from_VsDensity(2.7, 2.85)
    rho_m_8 = 2.85

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


def get_model_2layered():
    layer_1_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 7.3,
                      "optimize": True,
                      "min": 5,
                      "max": 10
                    },
					"Gm": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 0.1,
                      "max": 5
                    },
					"rho_m": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 1,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 0,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 3
                    },
                    "rho_f": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.01,
                      "max": 3
                    },
                    "phi": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })
    layer_2_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 21.5,
                      "optimize": True,
                      "min": 20,
                      "max": 25
                    },
					"Gm": {
                      "value": 17.5,
                      "optimize": False,
                      "min": 15,
                      "max": 20
                    },
					"rho_m": {
                      "value": 2.8,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 0.85,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 9.2,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0.4,
                      "optimize": False,
                      "min": 0.1,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 2.43,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0.05,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 1.8,
                      "optimize": False,
                      "min": 1,
                      "max": 2
                    },
                    "rho_f": {
                      "value": 0.95,
                      "optimize": False,
                      "min": 0.01,
                      "max": 1
                    },
                    "phi": {
                      "value": 0.1,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })

    return layer_1_dict, layer_2_dict

def get_model_3layered():
    layer_1_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 7.3,
                      "optimize": True,
                      "min": 5,
                      "max": 10
                    },
					"Gm": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 0.1,
                      "max": 5
                    },
					"rho_m": {
                      "value": 2.71,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 1,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 0,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 3
                    },
                    "rho_f": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.01,
                      "max": 3
                    },
                    "phi": {
                      "value": 0,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })
    layer_2_dict = OrderedDict({
                  "name": "xu-payne",
                  "components": {
                    "Km": {
                      "value": 21.5,
                      "optimize": True,
                      "min": 20,
                      "max": 25
                    },
					"Gm": {
                      "value": 17.5,
                      "optimize": False,
                      "min": 15,
                      "max": 20
                    },
					"rho_m": {
                      "value": 2.8,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 0.85,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 9.2,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0.4,
                      "optimize": False,
                      "min": 0.1,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 2.43,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0.05,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 1.8,
                      "optimize": False,
                      "min": 1,
                      "max": 2
                    },
                    "rho_f": {
                      "value": 0.95,
                      "optimize": False,
                      "min": 0.01,
                      "max": 1
                    },
                    "phi": {
                      "value": 0.1,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }
                  }
                })

    layer_3_dict = OrderedDict({
        "name": "xu-payne",
                "components":
        {"Km": {
                      "value": 22,
                      "optimize": True,
                      "min": 20,
                      "max": 25
                    },
					"Gm": {
                      "value": 10.7,
                      "optimize": False,
                      "min": 5,
                      "max": 15
                    },
					"rho_m": {
                      "value": 2.85,
                      "optimize": False,
                      "min": 2.5,
                      "max": 3
                    },
                    "Vm": {
                      "value": 1,
                      "optimize": False,
                      "min": 0.7,
                      "max": 1
                    },
                    "Ks": {
                      "value": 9.2,
                      "optimize": False,
                      "min": 7,
                      "max": 14
                    },
					"Gs": {
                      "value": 0.4,
                      "optimize": False,
                      "min": 0.1,
                      "max": 0.8
                    },
					"rho_s": {
                      "value": 2.43,
                      "optimize": False,
                      "min": 2.2,
                      "max": 2.6
                    },
                    "phi_s": {
                      "value": 0.0,
                      "optimize": False,
                      "min": 0.00,
                      "max": 0.1
                    },
                    "Kf": {
                      "value": 1.8,
                      "optimize": False,
                      "min": 1,
                      "max": 2
                    },
                    "rho_f": {
                      "value": 0.95,
                      "optimize": False,
                      "min": 0.01,
                      "max": 1
                    },
                    "phi": {
                      "value": 0.1,
                      "optimize": False,
                      "min": 0.001,
                      "max": 0.2
                    }}})

    return layer_1_dict, layer_2_dict, layer_3_dict

def main():
    # Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    layer_1_dict, layer_2_dict = get_model_2layered()
    h = 500

    layer_1 = Layer1D(h,
                      rp_attribute=RockPhysicsAttribute(layer_1_dict["components"], layer_1_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    layer_2 = Layer1D(-1,
                      rp_attribute=RockPhysicsAttribute(layer_2_dict["components"], layer_2_dict["name"]),
                      seism_attribute=None,
                      opt=LayerOPT.RP)

    model = SeismicModel1D([layer_1, layer_2])

    nlayers = 2
    dx = 50
    nx = 200
    x_rec = [i*dx for i in range(1, nx)]

    time_mark_1 = time.time()

    # DEM(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
    #         display_stat=True, visualize_res=True,
    #         calc_rays_p=True, calc_rays_s=True,
    #         calc_reflection_p=True, calc_reflection_s=False
    #         )
    wavetypes = [
        OWT.PdPu
        # OWT.PdSVu
    ]
    forward_with_trace_calcing(model, x_rec, dt=3e-03, trace_len=1500, wavetypes=wavetypes,
            display_stat=True, visualize_res=False,
                               visualize_seismograms=True
            )

    time_mark_2 = time.time()

    print(f'Calculation time: {time_mark_2 - time_mark_1}')

def water_case_test():
    h = [
        67.5,
        22.5,
        40,
    ]

    layer_1_seism = {
        'vp': 1500,
        'vs': 0,
        'rho': 1000
    }

    layer_2_seism = {
        'vp': 1600,
        'vs': 200,
        'rho': 1300
    }

    layer_3_seism = {
        'vp': 2600,
        'vs': 1000,
        'rho': 2000
    }

    layer_4_seism = {
        'vp': 3000,
        'vs': 1200,
        'rho': 2200
    }

    layer_1 = Layer1D(h[0],
                      rp_attribute=None,
                      seism_attribute=SeismicAttribute(**layer_1_seism),
                      opt=LayerOPT.NO)
    layer_2 = Layer1D(h[1],
                      seism_attribute=SeismicAttribute(**layer_2_seism),
                      rp_attribute=None,
                      opt=LayerOPT.NO)
    layer_3 = Layer1D(h[2],
                      rp_attribute=None,
                      seism_attribute=SeismicAttribute(**layer_3_seism),
                      opt=LayerOPT.NO)
    layer_4 = Layer1D(-1,
                      rp_attribute=None,
                      seism_attribute=SeismicAttribute(**layer_4_seism),
                      opt=LayerOPT.NO)

    model = SeismicModel1D([layer_1, layer_2, layer_3, layer_4])

    dx = 2
    nx = 100
    x_rec = [i * dx for i in range(1, nx + 1)]
    wave_types = [OWT.PdPu_water]

    start_time = time.time()
    observe, test_seismic = \
        forward_with_trace_calcing(model, x_rec,
                                   dt=1e-04, trace_len=2000, wavetypes=wave_types, display_stat=True,
                                   visualize_res=False, visualize_seismograms=True)
    end_time = time.time()

    print(end_time - start_time)


if __name__ == '__main__':
    # main()
    water_case_test()


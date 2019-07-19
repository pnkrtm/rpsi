import numpy as np
from matplotlib import pyplot as plt

from Objects.Models.Models import SeismicModel1D
from ForwardModeling.Seismic.Dynamic.ZoeppritzCoeffs import pdownpup, pdownpdown, puppup, svdownsvup, svdownsvdown, svupsvup
from ForwardModeling.Seismic.Dynamic.Reflection import zoeppritz_element
from Objects.Seismic.Observation import Observation, Source, Receiver
from Objects.Data.RDPair import OWT
from ForwardModeling.Seismic.RayTracing.Forward1DTracing import calculate_rays


def reflections_p_identity_1():
    vp = np.array([2500, 2700])
    vs = np.array([1000, 1300])
    rho = np.array([2710, 2597])
    h = np.array([500], 'int')
    dx = 30
    nrec = 50
    start_rec = 1

    model = SeismicModel1D(vp, vs, rho, h)
    depths = model.get_depths()

    sources = [Source(0, 0, 0)]
    receivers = [Receiver(x * dx, 0, 0) for x in range(start_rec, start_rec + nrec)]
    observe = Observation(sources, receivers)

    wavetype = OWT.PdPu
    rays_p = calculate_rays(observe, model, OWT.PdPu)

    # индекс интересующей отражающей границы
    refl_index = 2

    angles = np.array([r.get_reflection_angle() for r in rays_p[1]])

    # curve_1 = pdownpup(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)
    # curve_1 = puppup(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)
    # curve_1 = pdownpdown(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)
    # curve_1 = svdownsvup(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)
    # curve_1 = svdownsvdown(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)
    curve_1 = svupsvup(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles)

    # curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, angtype="theta", index=1, element='PdPu')
    # curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, angtype="theta", index=2, element="PuPu")
    # curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, element='PdPd')
    # curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, angtype="phi", index=1, element='SdSu')
    # curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, angtype="phi", index=1,
    #                             element='SdSd')
    curve_2 = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], angles, angtype="phi", index=2, element='SuSu')

    ampl1 = np.array([a.real for a in curve_1])
    ampl2 = np.array([a.real for a in curve_2])

    print(ampl1 - ampl2)

    plt.plot(ampl1)
    plt.plot(ampl2)
    plt.show()


if __name__ == '__main__':
    reflections_p_identity_1()

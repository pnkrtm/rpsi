from fmodeling.seismic.dynamic.reflection import zoeppritz_element
from fmodeling.seismic.dynamic.zoeppritz_coeffs import pdownpdown, puppup, pdownpup

def main():
    vp = [2006.75, 3227.5, 4495.611]
    vs = [1000, 1952.64, 2695]
    rho = [2710, 2596.5, 2850]

    theta = [22.56, 38.11, 38.11]

    refr_down = pdownpdown(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], theta[0])
    refl = pdownpup(vp[1], vs[1], rho[1], vp[2], vs[2], rho[2], theta[1])
    refr_up = puppup(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], theta[2])

    print(refr_down, refl, refr_up)

    refr_down_z = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], theta[0], element='PdPd')
    refl_z = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], theta[0], element='PdPu')
    refr_up_z = zoeppritz_element(vp[0], vs[0], rho[0], vp[1], vs[1], rho[1], theta[2], element='PuPu')

    print(refr_down_z, refl_z, refr_up_z)

def time_calculation():
    vp = [2006.75, 3227.5, 4495.611]
    vs = [1000, 1952.64, 2695]
    rho = [2710, 2596.5, 2850]

    theta = [22.56, 38.11, 38.11]

    import time
    t1 = time.time()
    refl = pdownpup(vp[1], vs[1], rho[1], vp[2], vs[2], rho[2], [theta[1]]*1000)
    t2 = time.time()
    print(f"Time evaluation: {t2 - t1}")


if __name__ == '__main__':
    # main()
    time_calculation()
import numpy as np

"""
Expressions for reflection/refraction coefficients in case of liquid-solid boundary

Based on article: 
"Analytical study of the reflection and transmission coefficientof the submarine interface"
Guangli Zhang, Chongtao Hao, Chen Yao
10 May 2018
"""


def pdownpup_water(vp1, rho1, vp2, vs2, rho2, theta1: np.ndarray=0):
    theta1 = np.radians(theta1).astype(complex) # p-wave angle

    multiple_angles = False
    if hasattr(theta1, 'shape') and len(theta1.shape) > 1 and theta1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta1 = theta1.T

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    cosphi2 = np.cos(phi2)
    costheta1 = np.cos(theta1)
    costheta2 = np.cos(theta2)

    a = vp1*vs2*rho1*rho2
    b = -vp2*vs2*rho2*rho2 + 4*p*p*vp2*vs2**3*rho2**2-4*p*p*costheta2*cosphi2*vs2**4*rho2*rho2 - 4*p**4*vp2*vs2*5*rho2**2
    # c = 2*p*p*vp1*vs2**3*rho1*rho2
    # d = 2*p*np.cos(theta2)*vs2*vs2*rho2
    # E = b*np.cos(theta1) - a*np.cos(theta2)
    # F = p*vp2*(c / (rho1*vp1) - rho2*vs2)

    rpp = (b*costheta1 + a*costheta2) / (b*costheta1 - a*costheta2)

    if multiple_angles:
        rpp = rpp.T

    return rpp


def pdownpdown_water(vp1, rho1, vp2, vs2, rho2, theta1: np.ndarray=0):
    theta1 = np.radians(theta1).astype(complex)  # p-wave angle

    multiple_angles = False
    if hasattr(theta1, 'shape') and len(theta1.shape) > 1 and theta1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta1 = theta1.T

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    cosphi2 = np.cos(phi2)
    costheta1 = np.cos(theta1)
    costheta2 = np.cos(theta2)

    a = vp1 * vs2 * rho1 * rho2
    b = -vp2 * vs2 * rho2 * rho2 + 4 * p * p * vp2 * vs2 ** 3 * rho2 ** 2 - 4 * p * p * costheta2 * cosphi2 * vs2 ** 4 * rho2 * rho2 -\
        4 * p ** 4 * vp2 * vs2 * 5 * rho2 ** 2
    c = 2 * p * p * vp1 * vs2 ** 3 * rho1 * rho2
    # d = 2 * p * np.cos(theta2) * vs2 * vs2 * rho2
    # E = b * np.cos(theta1) - a * np.cos(theta2)
    # F = p * vp2 * (c / (rho1 * vp1) - rho2 * vs2)

    tpp = (2*costheta1*(c-a) / (b*costheta1 - a*costheta2)) * np.sqrt((rho2*vp2*costheta2) / (rho1*vp1*costheta1))

    if multiple_angles:
        tpp = tpp.T

    return tpp


def puppup_water(vp1, rho1, vp2, vs2, rho2, theta1: np.ndarray=0):
    theta1 = np.radians(theta1).astype(complex)  # p-wave angle

    multiple_angles = False
    if hasattr(theta1, 'shape') and len(theta1.shape) > 1 and theta1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta1 = theta1.T

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    cosphi2 = np.cos(phi2)
    costheta1 = np.cos(theta1)
    costheta2 = np.cos(theta2)

    a = vp1 * vs2 * rho1 * rho2
    b = -vp2 * vs2 * rho2 * rho2 + 4 * p * p * vp2 * vs2 ** 3 * rho2 ** 2 - 4 * p * p * costheta2 * cosphi2 * vs2 ** 4 * rho2 * rho2 - \
        4 * p ** 4 * vp2 * vs2 * 5 * rho2 ** 2
    c = 2 * p * p * vp1 * vs2 ** 3 * rho1 * rho2
    d = 2 * p * costheta2 * vs2 * vs2 * rho2
    E = b * costheta1 - a * costheta2
    F = p * vp2 * (c / (rho1 * vp1) - rho2 * vs2)

    tpp = (b*costheta2 + (F + d*costheta2)*2*rho2*vs2*vs2*p*costheta2 +(2*vs2*vs2*p*p - 1)*rho2*vp2*rho2*vs2*costheta2)/E * \
          np.sqrt((rho1 * vp1 * costheta1) / (rho2 * vp2 * costheta2))

    if multiple_angles:
        tpp = tpp.T

    return tpp

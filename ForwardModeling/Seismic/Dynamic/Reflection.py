# -*- coding: utf-8 -*-
""""
"""
from collections import namedtuple
from Objects.Seismic.Rays import BoundaryType
from Objects.Data.WavePlaceholder import OWT
from ForwardModeling.Seismic.Dynamic.ZoeppritzCoeffs import pdownpup, svdownsvup
from utils.vectorizing import vectorize

import numpy as np

# from bruges.rockphysics import moduli
# from bruges.rockphysics import anisotropy
# from bruges.util import deprecated


def critical_angles(vp1, vp2, vs2=None):
    """
    Compute critical angle at an interface, given the upper (vp1) and
    lower (vp2) velocities. If you want the PS-wave critical angle as well,
    pass vs2 as well.
    Args:
        vp1 (ndarray): The upper layer P-wave velocity.
        vp2 (ndarray): The lower layer P-wave velocity.
    Returns:
        tuple: The first and second critical angles at the interface, in
            degrees. If there isn't a critical angle, it is set to np.nan.
    """
    ca1 = ca2 = np.nan

    if vp1 > vp2:
        ca1 = np.degrees(np.arcsin(vp1/vp2))

    if (vs2 is not None) and (vp1 > vs2):
        ca2 = np.degrees(np.arcsin(vp1/vs2))

    return ca1, ca2


def reflection_phase(reflectivity):
    """
    Compute the phase of the reflectivity. Returns an array (or float) of
    the phase, in positive multiples of 180 deg or pi rad. So 1 is opposite
    phase. A value of 1.1 would be +1.1 \times \pi rad.
    Args:
        reflectivity (ndarray): The reflectivity, eg from `zoeppritz()`.
    Returns:
        ndarray: The phase, strictly positive
    """
    ph = np.arctan2(np.imag(reflectivity), np.real(reflectivity)) / np.pi
    ph[ph == 1] = 0
    ph[ph < 0] = 2 + ph[ph < 0]
    return ph


def acoustic_reflectivity(vp, rho):
    """
    The acoustic reflectivity, given Vp and RHOB logs.
    Args:
        vp (ndarray): The P-wave velocity.
        rho (ndarray): The bulk density.
    Returns:
        ndarray: The reflectivity coefficient series.
    """
    upper = vp[:-1] * rho[:-1]
    lower = vp[1:] * rho[1:]
    return (lower - upper) / (lower + upper)


def reflectivity(vp, vs, rho, theta=0, method='zoeppritz_rpp'):
    """
    Offset reflectivity, given Vp, Vs, rho, and offset.
    Computes 'upper' and 'lower' intervals from the three provided arrays,
    then passes the result to the specified method to compute reflection
    coefficients.
    For acoustic reflectivity, either use the `acoustic_reflectivity()`
    function, or call `reflectivity()` passing any log as Vs, e.g. just give
    the Vp log twice (it won't be used anyway):
        reflectivity(vp, vp, rho)
    For anisotropic reflectivity, use either `anisotropy.blangy()` or
    `anisotropy.ruger()`.
    Args:
        vp (ndarray): The P-wave velocity; float or 1D array length m.
        vs (ndarray): The S-wave velocity; float or 1D array length m.
        rho (ndarray): The density; float or 1D array length m.
        theta (ndarray): The incidence angle; float or 1D array length n.
        method (str): The reflectivity equation to use; one of:
                - 'scattering_matrix': scattering_matrix
                - 'zoeppritz_element': zoeppritz_element
                - 'zoeppritz': zoeppritz
                - 'zoeppritz_rpp': zoeppritz_rpp
                - 'akirichards': akirichards
                - 'akirichards_alt': akirichards_alt
                - 'fatti': fatti
                - 'shuey': shuey
                - 'bortfeld': bortfeld
                - 'hilterman': hilterman
        Notes:
                - scattering_matrix gives the full solution
                - zoeppritz_element gives a single element which you specify
                - zoeppritz returns RPP element only; use zoeppritz_rpp instead
                - zoeppritz_rpp is faster than zoeppritz or zoeppritz_element
    Returns:
        ndarray. The result of running the specified method on the inputs.
            Will be a float (for float inputs and one angle), a 1 x n array
            (for float inputs and an array of angles), a 1 x m-1 array (for
            float inputs and one angle), or an m-1 x n array (for array inputs
            and an array of angles).
    """
    methods = {
        'scattering_matrix': scattering_matrix,
        'zoeppritz_element': zoeppritz_element,
        'zoeppritz': zoeppritz,
        'zoeppritz_rpp': pdownpup,
        'akirichards': akirichards,
        'akirichards_alt': akirichards_alt,
        # 'fatti': fatti,
        # 'shuey': shuey,
        # 'bortfeld': bortfeld,
        # 'hilterman': hilterman,
    }
    func = methods[method.lower()]
    vp = np.asanyarray(vp, dtype=float)
    vs = np.asanyarray(vs, dtype=float)
    rho = np.asanyarray(rho, dtype=float)

    vp1, vp2 = vp[:-1], vp[1:]
    vs1, vs2 = vs[:-1], vs[1:]
    rho1, rho2 = rho[:-1], rho[1:]

    return func(vp1, vs1, rho1, vp2, vs2, rho2, theta)


def angles_definition(vp1, vs1, vp2, vs2, angle=0, angtype="theta", index=1):
    if angtype == "theta" and index == 1:
        theta1 = angle
        theta1 = np.radians(theta1).astype(complex) * np.ones_like(vp1)
        p = np.sin(theta1) / vp1  # Ray parameter.

    elif angtype == "theta" and index == 2:
        theta2 = angle
        theta2 = np.radians(theta2).astype(complex) * np.ones_like(vp2)
        p = np.sin(theta2) / vp2  # Ray parameter.

    elif angtype == "phi" and index == 1:
        phi1 = angle
        phi1 = np.radians(phi1).astype(complex) * np.ones_like(vs1)
        p = np.sin(phi1) / vs1  # Ray parameter.

    elif angtype == "phi" and index == 2:
        phi2 = angle
        phi2 = np.radians(phi2).astype(complex) * np.ones_like(vs2)
        p = np.sin(phi2) / vs2  # Ray parameter.

    else:
        raise ValueError()

    theta1 = np.arcsin(p * vp1)  # Refl. angle of P-wave.
    theta2 = np.arcsin(p * vp2)  # Trans. angle of P-wave.
    phi1 = np.arcsin(p * vs1)  # Refl. angle of converted S-wave.
    phi2 = np.arcsin(p * vs2)  # Trans. angle of converted S-wave.

    return theta1, theta2, phi1, phi2

# @vectorize
def scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2, angle=0, angtype="theta", index=1):
    """
    Full Zoeppritz solution, considered the definitive solution.
    Calculates the angle dependent p-wave reflectivity of an interface
    between two mediums.
    Originally written by: Wes Hamlyn, vectorized by Agile.
    Returns the complex reflectivity.
    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
    Returns:
        ndarray. The exact Zoeppritz solution for all modes at the interface.
            A 4x4 array representing the scattering matrix at the incident
            angle theta1.
    """
    theta1, theta2, phi1, phi2 = angles_definition(vp1, vs1, vp2, vs2, angle, angtype, index)

    # Matrix form of Zoeppritz equations... M & N are matrices.
    M = np.array([[-np.sin(theta1), -np.cos(phi1), np.sin(theta2), np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [-rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   rho1 * vs1 * np.sin(2 * phi1),
                   rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   -rho2 * vs2 * np.sin(2 * phi2)]])

    N = np.array([[np.sin(theta1), np.cos(phi1), -np.sin(theta2), -np.cos(phi2)],
                  [np.cos(theta1), -np.sin(phi1), np.cos(theta2), -np.sin(phi2)],
                  [2 * rho1 * vs1 * np.sin(phi1) * np.cos(theta1),
                   rho1 * vs1 * (1 - 2 * np.sin(phi1) ** 2),
                   2 * rho2 * vs2 * np.sin(phi2) * np.cos(theta2),
                   rho2 * vs2 * (1 - 2 * np.sin(phi2) ** 2)],
                  [rho1 * vp1 * (1 - 2 * np.sin(phi1) ** 2),
                   -rho1 * vs1 * np.sin(2 * phi1),
                   - rho2 * vp2 * (1 - 2 * np.sin(phi2) ** 2),
                   rho2 * vs2 * np.sin(2 * phi2)]])

    M_ = np.moveaxis(np.squeeze(M), [0, 1], [-2, -1])
    A = np.linalg.inv(M_)
    N_ = np.moveaxis(np.squeeze(N), [0, 1], [-2, -1])
    Z_ = np.matmul(A, N_)

    return np.transpose(Z_, axes=list(range(Z_.ndim - 2)) + [-1, -2])


def zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, angle=0, angtype="theta", index=1, element='PdPu'):
    """
    Returns any mode reflection coefficients from the Zoeppritz
    scattering matrix. Pass in the mode as element, e.g. 'PdSu' for PS.
    Wraps scattering_matrix().
    Returns the complex reflectivity.
    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        element (str): The name of the element to return, must be one of:
            'PdPu', 'SdPu', 'PuPu', 'SuPu', 'PdSu', 'SdSu', 'PuSu', 'SuSu',
            'PdPd', 'SdPd', 'PuPd', 'SuPd', 'PdSd', 'SdSd', 'PuSd', 'SuSd'.
    Returns:
        ndarray. Array length n of the exact Zoeppritz solution for the
            specified modes at the interface, at the incident angle theta1.
    """
    elements = np.array([['PdPu', 'SdPu', 'PuPu', 'SuPu'],
                         ['PdSu', 'SdSu', 'PuSu', 'SuSu'],
                         ['PdPd', 'SdPd', 'PuPd', 'SuPd'],
                         ['PdSd', 'SdSd', 'PuSd', 'SuSd']])

    Z = scattering_matrix(vp1, vs1, rho1, vp2, vs2, rho2,  angle, angtype, index).T

    return np.squeeze(Z[np.where(elements == element)].T)


def zoeppritz(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Returns the PP reflection coefficients from the Zoeppritz
    scattering matrix. Wraps zoeppritz_element().
    Returns the complex reflectivity.
    Args:
        vp1 (float): The upper P-wave velocity.
        vs1 (float): The upper S-wave velocity.
        rho1 (float): The upper layer's density.
        vp2 (float): The lower P-wave velocity.
        vs2 (float): The lower S-wave velocity.
        rho2 (float): The lower layer's density.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
    Returns:
        ndarray. Array length n of the exact Zoeppritz solution for the
            specified modes at the interface, at the incident angle theta1.
    """
    return zoeppritz_element(vp1, vs1, rho1, vp2, vs2, rho2, theta1, 'PdPu')


@vectorize
def akirichards(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    The Aki-Richards approximation to the reflectivity.
    This is the formulation from Avseth et al., _Quantitative seismic
    interpretation_, Cambridge University Press, 2006. Adapted for a 4-term
    formula. See http://subsurfwiki.org/wiki/Aki-Richards_equation.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.
    Returns:
        ndarray. The Aki-Richards approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.radians(theta1).astype(complex)

    # critical_angle = arcsin(vp1/vp2)
    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    meantheta = (theta1+theta2) / 2.0
    rho = (rho1+rho2) / 2.0
    vp = (vp1+vp2) / 2.0
    vs = (vs1+vs2) / 2.0

    # Compute the coefficients
    w = 0.5 * drho/rho
    x = 2 * (vs/vp1)**2 * drho/rho
    y = 0.5 * (dvp/vp)
    z = 4 * (vs/vp1)**2 * (dvs/vs)

    # Compute the terms
    term1 = w
    term2 = -1 * x * np.sin(theta1)**2
    term3 = y / np.cos(meantheta)**2
    term4 = -1 * z * np.sin(theta1)**2

    if terms:
        fields = ['term1', 'term2', 'term3', 'term4']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(np.squeeze([term1 for _ in theta1]),
                           np.squeeze(term2),
                           np.squeeze(term3),
                           np.squeeze(term4)
                           )
    else:
        return np.squeeze(term1 + term2 + term3 + term4)


@vectorize
def akirichards_alt(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, terms=False):
    """
    This is another formulation of the Aki-Richards solution.
    See http://subsurfwiki.org/wiki/Aki-Richards_equation
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence angle; float or 1D array length n.
        terms (bool): Whether or not to return a tuple of the terms of the
            equation. The first term is the acoustic impedance.
    Returns:
        ndarray. The Aki-Richards approximation for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.radians(theta1).astype(complex)

    # critical_angle = arcsin(vp1/vp2)
    theta2 = np.arcsin(vp2/vp1*np.sin(theta1))
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    theta = (theta1+theta2)/2.0
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute the three terms
    term1 = 0.5 * (dvp/vp + drho/rho)
    term2 = (0.5*dvp/vp-2*(vs/vp)**2*(drho/rho+2*dvs/vs)) * np.sin(theta)**2
    term3 = 0.5 * dvp/vp * (np.tan(theta)**2 - np.sin(theta)**2)

    if terms:
        fields = ['term1', 'term2', 'term3']
        AkiRichards = namedtuple('AkiRichards', fields)
        return AkiRichards(np.squeeze([term1 for _ in theta1]),
                           np.squeeze(term2),
                           np.squeeze(term3)
                           )
    else:
        return np.squeeze(term1 + term2 + term3)


def calculate_reflections_vectorized(model, rays, element):
    """
    Calculating reflections with vectorized objects
    :param model:
    :param rays: dictionary { boundary_index: reflected_rays }
    :param element:
    :param calculate_refraction_flag:
    :return:
    """
    depths = model.get_depths()

    angles_all = []
    offsets_all = []

    for i, d in enumerate(depths[1:], 1):
        # check if boundary is reflected
        if i in rays.keys():

            depth_rays = rays[i]
            angles = [r.get_reflection_angle() for r in depth_rays]
            offsets = [r.x_finish for r in depth_rays]

            angles = np.rad2deg(angles)

            angles_all.append(angles)
            offsets_all.append(offsets)

    angles_all = np.array(angles_all)
    offsets_all = np.array(offsets_all)

    # Берем все границы кроме последней
    vp1_arr = model.get_single_param(param_name='vp', index_finish=-1)
    vs1_arr = model.get_single_param(param_name='vs', index_finish=-1)
    rho1_arr = model.get_single_param(param_name='rho', index_finish=-1)

    # Берем все границы кроме первой
    vp2_arr = model.get_single_param(param_name='vp', index_start=1)
    vs2_arr = model.get_single_param(param_name='vs', index_start=1)
    rho2_arr = model.get_single_param(param_name='rho', index_start=1)

    refection_indexes = np.array(list(rays.keys())) - 1

    vp1_arr = vp1_arr[refection_indexes]
    vs1_arr = vs1_arr[refection_indexes]
    rho1_arr = rho1_arr[refection_indexes]

    vp2_arr = vp2_arr[refection_indexes]
    vs2_arr = vs2_arr[refection_indexes]
    rho2_arr = rho2_arr[refection_indexes]

    if element == OWT.PdPu:
        reflection_amplitudes = pdownpup(vp1_arr, vs1_arr, rho1_arr,
                                              vp2_arr, vs2_arr, rho2_arr, angles_all)

    elif element == OWT.SVdSVu:
        reflection_amplitudes = svdownsvup(vp1_arr, vs1_arr, rho1_arr,
                                              vp2_arr, vs2_arr, rho2_arr, angles_all)

    else:
        raise NotImplementedError("Another reflection types are not implemented yet!")

    return reflection_amplitudes


def calculate_reflections(model, rays, owt):
    """

    :param model:
    :param rays: dictionary {boundary_index: reflected rays}
    :param owt: observation wave type
    :return:
    """
    reflections_amplitudes = calculate_reflections_vectorized(model, rays, owt)

    i = 1
    for dr, da in zip(rays.values(), reflections_amplitudes):
        for ofr, ofa in zip(dr, da):
            ofr.add_boundary_dynamic(ofa, BoundaryType.REFLECTION, i)

        i += 1

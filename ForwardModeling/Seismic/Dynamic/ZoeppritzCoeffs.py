import numpy as np


def pdownpup(vp1, vs1, rho1, vp2, vs2, rho2, theta1: np.ndarray=0 ):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence P-wave angle; float or 1D array length n or 2D array with shape (m, n).
    Returns:
        ndarray. The exact Zoeppritz solution for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.radians(theta1).astype(complex)

    multiple_angles = False
    if hasattr(theta1, 'shape') and len(theta1.shape) > 1 and theta1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta1 = theta1.T

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2)**2.) - rho1 * (1 - 2 * np.sin(phi1)**2.)
    b = rho2 * (1 - 2 * np.sin(phi2)**2.) + 2 * rho1 * np.sin(phi1)**2.
    c = rho1 * (1 - 2 * np.sin(phi1)**2.) + 2 * rho2 * np.sin(phi2)**2.
    d = 2 * (rho2 * vs2**2 - rho1 * vs1**2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1)/vp1 * np.cos(phi2)/vs2
    H = a - d * np.cos(theta2)/vp2 * np.cos(phi1)/vs1

    D = E*F + G*H*p**2

    rpp = (1/D) * (F*(b*(np.cos(theta1)/vp1) - c*(np.cos(theta2)/vp2)) \
                   - H*p**2 * (a + d*(np.cos(theta1)/vp1)*(np.cos(phi2)/vs2)))

    if multiple_angles:
        rpp = rpp.T

    return rpp


def pdownpdown(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta1 (ndarray): The incidence P-wave angle; float or 1D array length n or 2D array with shape (m, n). IN DEGREES!!
    Returns:
        ndarray. The exact Zoeppritz solution for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta1 = np.radians(theta1).astype(complex)

    multiple_angles = False
    if hasattr(theta1, 'shape') and len(theta1.shape) > 1 and theta1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta1 = theta1.T

    p = np.sin(theta1) / vp1  # Ray parameter
    theta2 = np.arcsin(p * vp2)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2.) - rho1 * (1 - 2 * np.sin(phi1) ** 2.)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2.) + 2 * rho1 * np.sin(phi1) ** 2.
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2.) + 2 * rho2 * np.sin(phi2) ** 2.
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    rpp = (2 * rho1 * (np.cos(theta1) / vp1) * F * vp1) / (vp2 * D)

    if multiple_angles:
        rpp = rpp.T

    return rpp


def puppup(vp1, vs1, rho1, vp2, vs2, rho2, theta2=0):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        theta2 (ndarray): The incidence P-wave angle; float or 1D array length n  or 2D array with shape (m, n).
    Returns:
        ndarray. The exact Zoeppritz solution for P-P reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """
    theta2 = np.radians(theta2).astype(complex)

    multiple_angles = False
    if hasattr(theta2, 'shape') and len(theta2.shape) > 1 and theta2.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        theta2 = theta2.T

    p = np.sin(theta2) / vp2  # Ray parameter
    theta1 = np.arcsin(p * vp1)
    phi1 = np.arcsin(p * vs1)  # Reflected S
    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2.) - rho1 * (1 - 2 * np.sin(phi1) ** 2.)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2.) + 2 * rho1 * np.sin(phi1) ** 2.
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2.) + 2 * rho2 * np.sin(phi2) ** 2.
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    rpp = (2 * rho2 * (np.cos(theta2) / vp2) * F * vp2) / (vp1* D)

    if multiple_angles:
        rpp = rpp.T

    return rpp


def svdownsvup(vp1, vs1, rho1, vp2, vs2, rho2, phi1=0):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        phi1 (ndarray): The incidence S-wave angle; float or 1D array length n or 2D array with shape (m, n).
    Returns:
        ndarray. The exact Zoeppritz solution for Sv-Sv reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """

    phi1 = np.radians(phi1).astype(complex)

    multiple_angles = False
    if hasattr(phi1, 'shape') and len(phi1.shape) > 1 and phi1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        phi1 = phi1.T

    p = np.sin(phi1) / vs1  # Ray parameter

    theta1 = np.arcsin(p * vp1) # Reflected P
    theta2 = np.arcsin(p * vp2) # Transmitted P

    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2.) - rho1 * (1 - 2 * np.sin(phi1) ** 2.)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2.) + 2 * rho1 * np.sin(phi1) ** 2.
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2.) + 2 * rho2 * np.sin(phi2) ** 2.
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    coeff_1 = b * np.cos(phi1) / vs1 - c * np.cos(phi2) / vs2
    coeff_2 = a + d * (np.cos(theta2) / vp2) * (np.cos(phi1) / vs1)

    # rsvsv = -((b * np.cos(phi1) / vs1 - c * np.cos(phi2) / vs2) * E
    #           - (a + d * (np.cos(theta2) / vp2) * (np.cos(phi1) / vs1)) * G * p * p) / D

    rsvsv = -(coeff_1 * E - coeff_2 * G * p * p) / D

    if multiple_angles:
        rsvsv = rsvsv.T

    return rsvsv


def svdownsvdown(vp1, vs1, rho1, vp2, vs2, rho2, phi1=0):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        phi1 (ndarray): The incidence S-wave angle; float or 1D array length n or 2D array with shape (m, n).
    Returns:
        ndarray. The exact Zoeppritz solution for Sv-Sv reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """

    phi1 = np.radians(phi1).astype(complex)

    multiple_angles = False
    if hasattr(phi1, 'shape') and len(phi1.shape) > 1 and phi1.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        phi1 = phi1.T

    p = np.sin(phi1) / vs1  # Ray parameter

    theta1 = np.arcsin(p * vp1) # Reflected P
    theta2 = np.arcsin(p * vp2) # Transmitted P

    phi2 = np.arcsin(p * vs2)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2.) - rho1 * (1 - 2 * np.sin(phi1) ** 2.)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2.) + 2 * rho1 * np.sin(phi1) ** 2.
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2.) + 2 * rho2 * np.sin(phi2) ** 2.
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    rsvsv = 2 * rho1 * (np.cos(phi1) / vs1) * E * vs1 / (vs2 * D)

    if multiple_angles:
        rsvsv = rsvsv.T

    return rsvsv


def svupsvup(vp1, vs1, rho1, vp2, vs2, rho2, phi2=0):
    """
    Exact Zoeppritz from expression.
    This is useful because we can pass arrays to it, which we can't do to
    scattering_matrix().
    Dvorkin et al. (2014). Seismic Reflections of Rock Properties. Cambridge.
    Returns the complex reflectivity.
    Args:
        vp1 (ndarray): The upper P-wave velocity; float or 1D array length m.
        vs1 (ndarray): The upper S-wave velocity; float or 1D array length m.
        rho1 (ndarray): The upper layer's density; float or 1D array length m.
        vp2 (ndarray): The lower P-wave velocity; float or 1D array length m.
        vs2 (ndarray): The lower S-wave velocity; float or 1D array length m.
        rho2 (ndarray): The lower layer's density; float or 1D array length m.
        phi1 (ndarray): The incidence S-wave angle; float or 1D array length n or 2D array with shape (m, n).
    Returns:
        ndarray. The exact Zoeppritz solution for Sv-Sv reflectivity at the
            interface. Will be a float (for float inputs and one angle), a
            1 x n array (for float inputs and an array of angles), a 1 x m
            array (for float inputs and one angle), or an n x m array (for
            array inputs and an array of angles).
    """

    phi2 = np.radians(phi2).astype(complex)

    multiple_angles = False
    if hasattr(phi2, 'shape') and len(phi2.shape) > 1 and phi2.shape[1] > 1:
        multiple_angles = True

    if multiple_angles:
        phi2 = phi2.T

    p = np.sin(phi2) / vs2  # Ray parameter

    theta1 = np.arcsin(p * vp1) # Reflected P
    theta2 = np.arcsin(p * vp2) # Transmitted P

    phi1 = np.arcsin(p * vs1)  # Transmitted S

    a = rho2 * (1 - 2 * np.sin(phi2) ** 2.) - rho1 * (1 - 2 * np.sin(phi1) ** 2.)
    b = rho2 * (1 - 2 * np.sin(phi2) ** 2.) + 2 * rho1 * np.sin(phi1) ** 2.
    c = rho1 * (1 - 2 * np.sin(phi1) ** 2.) + 2 * rho2 * np.sin(phi2) ** 2.
    d = 2 * (rho2 * vs2 ** 2 - rho1 * vs1 ** 2)

    E = (b * np.cos(theta1) / vp1) + (c * np.cos(theta2) / vp2)
    F = (b * np.cos(phi1) / vs1) + (c * np.cos(phi2) / vs2)
    G = a - d * np.cos(theta1) / vp1 * np.cos(phi2) / vs2
    H = a - d * np.cos(theta2) / vp2 * np.cos(phi1) / vs1

    D = E * F + G * H * p ** 2

    rsvsv = 2 * rho2 * (np.cos(phi2) / vs2) * E * vs2 / (vs1 * D)

    if multiple_angles:
        rsvsv = rsvsv.T

    return rsvsv
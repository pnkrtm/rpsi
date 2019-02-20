from functools import wraps
import numpy as np

def vectorize(func):
    """
    Decorator to make sure the inputs are arrays. We also add a dimension
    to theta to make the functions work in an 'outer product' way.
    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta, plus kwargs. Returns
    that function with the arguments transformed to ndarrays.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):
        vp1 = np.asanyarray(vp1, dtype=float)
        vs1 = np.asanyarray(vs1, dtype=float) + 1e-12  # Prevent singular matrix.
        rho1 = np.asanyarray(rho1, dtype=float)
        vp2 = np.asanyarray(vp2, dtype=float)
        vs2 = np.asanyarray(vs2, dtype=float) + 1e-12  # Prevent singular matrix.
        rho2 = np.asanyarray(rho2, dtype=float)
        theta1 = np.asanyarray(theta1, dtype=float)
        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper
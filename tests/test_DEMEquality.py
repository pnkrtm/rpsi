import time
import numpy as np

from fmodeling.rock_physics.mediums.DEM import DEM
from fmodeling.rock_physics.mediums.DEMSlb import DEM as DEMSlb

def main():
    # Matrix properties
    Km = 77.0  # GPa
    Gm = 32.0  # GPa
    rhom = 2.71  # g/cm3

    # Fluid properties
    Kf = 3.0  # GPa
    rhof = 1.0  # g/cm3

    # Porosity
    phimax = 1

    phi = 0.1
    alpha = 0.1

    # Inclusion properties
    # In this example a mixture of three inclusion types are used:
    # - 30% of 0.02 aspect ratio
    # - 50% of 0.15 aspect ratio
    # - 20% of 0.80 aspect ratio
    alphas = np.array([0.01, 0.15, 0.8])
    volumes = np.array([0.3, 0.5, 0.2]) * phimax

    time_point_1 = time.time()
    Km_DEM, Gm_DEM, phi_DEM = DEM(Km, Gm, np.array([0]), np.array([0]), np.array([alpha]), np.array([phi*phimax]))
    time_point_2 = time.time()
    Km_DEMSlb, Gm_DEMSlb = DEMSlb(Km, Gm, 0.0, 0.0, phi, alpha, phimax)
    time_point_3 = time.time()

    print(Km_DEM, Gm_DEM, phi_DEM)
    print(Km_DEMSlb, Gm_DEMSlb)

    print(f"Old calculation time: {time_point_2 - time_point_1}")
    print(f"New calculation time: {time_point_3 - time_point_2}")


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt

from inversion.DataIO import create_start_stop_indexes
from fmodeling.ForwardProcessing1D import forward_with_trace_calcing
from Tests.test_ForwardProcessing1D import get_model_1
from Visualization.Seismic import visualize_seismogram


def main():
    Km, Gm, rho_m, Ks, Gs, rho_s, Kf, rho_f, phi, phi_s, h = get_model_1()
    nlayers = 8
    dx = 50
    nx = 20
    x_rec = [i * dx for i in range(1, nx)]
    dt = 3e-03

    observe, model, rays_p, rays_s, seismogram_p, seismogram_s = forward_with_trace_calcing(nlayers, Km, Gm, Ks, Gs, Kf, phi, phi_s, rho_s, rho_f, rho_m, h, x_rec,
                               dt=dt, trace_len=1500,
                               display_stat=True, visualize_res=False,
                               use_p_waves=True, use_s_waves=False,
                               visualize_seismograms=False
                               )

    indexes_params = {
      "type": "from_model",
      "values": {
        "start": {
          "v": 2600,
          "h": 500
        },
        "stop": {
          "v": 2000,
          "h": 1000
        }
      }
    }

    start_indexes, stop_indexes = create_start_stop_indexes(indexes_params, np.array(x_rec), dt)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(x_rec, start_indexes*dt)
    axes.plot(x_rec, stop_indexes*dt)

    visualize_seismogram(axes, seismogram_p, normalize=True, wiggles=False)
    axes.set_title('p-waves seismogram')

    plt.show()


if __name__ == '__main__':
    main()

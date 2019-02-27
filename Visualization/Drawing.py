from matplotlib import pyplot as plt

from Visualization.Seismic import visualize_seismogram


def draw_seismogram(seismogram, pic_header, filename, gain=1):
    visualize_seismogram(plt, seismogram, normalize=False, fill_negative=False, wigles=False, gain=gain)
    plt.title(pic_header, fontsize=18)
    plt.ylabel('time, s')
    plt.xlabel('distance, m')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

from matplotlib import pyplot as plt

from Visualization.Seismic import visualize_seismogram


def draw_seismogram(seismogram, pic_header, filename, gain=1,  additional_lines: list=None, colorbar: bool=False):
    fig, axes = plt.subplots(nrows=1, ncols=1)

    if additional_lines is not None:
        for al in additional_lines:
            axes.plot(al[0], al[1], **al[2])

    visualize_seismogram(axes, seismogram, normalize=False, fill_negative=False, wigles=False, gain=gain)
    plt.title(pic_header, fontsize=18)
    plt.ylabel('time, s')
    plt.xlabel('distance, m')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

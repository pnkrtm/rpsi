from matplotlib import pyplot as plt

from visualization.Seismic import visualize_seismogram


def draw_seismogram(seismogram, pic_header, filename, gain=1,  additional_lines: list=None, colorbar: bool=False,
                    fill_positive=False, wiggles=False, normalize=True):
    fig, axes = plt.subplots(nrows=1, ncols=1)

    if additional_lines is not None:
        for al in additional_lines:
            axes.plot(al[0], al[1], **al[2])

    visualize_seismogram(fig, axes, seismogram, normalize=normalize, fill_positive=fill_positive, wiggles=wiggles, gain=gain,
                         colorbar=colorbar)
    plt.title(pic_header, fontsize=18)
    plt.ylabel('time, s')
    plt.xlabel('distance, m')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def draw_dos_seismograms(seismogram1, seismogram2, pic_header, filename, gain=1, normalize=True):
    fig, axes = plt.subplots(nrows=1, ncols=1)

    visualize_seismogram(fig, axes, seismogram1, normalize=normalize, fill_positive=False, wiggles=True, gain=gain,
                         wiggle_color='k')
    visualize_seismogram(fig, axes, seismogram2, normalize=normalize, fill_positive=False, wiggles=True, gain=gain,
                         wiggle_color='r-')
    plt.title(pic_header, fontsize=18)
    plt.ylabel('time, s')
    plt.xlabel('distance, m')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

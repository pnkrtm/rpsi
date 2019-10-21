import numpy as np


def bgtl(Km, Gm, Kfl, phi):
    """
    Модель для рыхлого осадка с пористостью выше критической, заполненный некоторым осадком
    :param Km: Модуль сжатия сухой породы
    :param Gm: Модель сдвига сухой породы
    :param Kfl: Модуль сжатия флюида
    :param phi: Пористость
    :return:
    """
    A1 = -183.05186
    A2 = 0.99494

    beta = (A1 - A2) / (1 + np.exp((phi + 0.56468) / 0.10817)) + A2

    M = 1 / ((beta - phi) / Km + (phi / Kfl))

    K = Km * (1 - beta) + beta * beta * M

    G = (Gm * Km * (1 - beta) * (1 - phi) * (1 - phi) + Gm * beta * beta * M * (1 - phi) * (1 - phi)) / \
        (Km + (4*Gm*(1 - (1-phi)*(1-phi))) / 3)

    return K, G

import numpy as np


def voigt(m: np.ndarray, f: np.ndarray):
    '''
    Осреднение по Войгту
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''
    return np.sum(m*f)


def reuss(m: np.ndarray, f: np.ndarray):
    '''
    Осреднение по Рёссу
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''
    return 1/np.sum(f/m)


def hill(m: np.ndarray, f: np.ndarray):
    '''
    Осреднение по Рёссу
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''

    return (voigt(m, f) + reuss(m, f)) / 2

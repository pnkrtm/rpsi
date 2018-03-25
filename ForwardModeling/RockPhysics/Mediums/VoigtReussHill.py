import numpy as np


def voigt(m, f):
    '''
    Осреднение по Войгту
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''
    return np.sum(m*f)


def reuss(m, f):
    '''
    Осреднение по Рёссу
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''
    return 1/np.sum(f/m)


def hill(m, f):
    '''
    Осреднение по Рёссу
    :param m: Массив K или G
    :param f: Массив объемных долей
    :return:
    '''

    return (voigt(m, f) + reuss(m, f)) / 2

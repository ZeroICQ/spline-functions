import numpy as np
import matplotlib.pyplot as plt
import sympy
import sympy.functions
from utils import *

x_start = 0
x_end = 7
step = 0.2
a = 1


def f_symbol():
    x = sympy.Symbol('x')
    return sympy.functions.sin(a*x)


def f_real(xf):
    x = sympy.Symbol('x')
    return f_symbol().subs(x, xf).evalf()


def h_i(i, xes):
    return xes[i+1] - xes[i]


def mu_i(i, xes):
    return h_i(i-1, xes) / (h_i(i-1, xes) + h_i(i, xes))


def lambda_i(i, xes):
    return 1 - mu_i(i, xes)


def main():
    xes = np.arange(x_start, x_end, step)
    fs = [f_real(x) for x in xes]
    plt.plot(xes, fs, 'bo', label="Табличная функция f(x)")
    high_density_xes = np.arange(x_start, x_end, 0.01)

    real_fs = [f_real(x) for x in high_density_xes]
    plt.plot(high_density_xes, real_fs, 'r,-', label='f(x)')

    prepare_plot()
    plt.show()


if __name__ == '__main__':
    main()

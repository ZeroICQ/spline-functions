import numpy as np
import matplotlib.pyplot as plt
import sympy
import sympy.functions
import sympy.abc
from utils import *

x_start = 0
a = 1
x_end = 4 * (2 * math.pi / a) + 0.1
step = 0.2


def f_symbol():
    x = sympy.Symbol('x')
    return sympy.functions.sin(a*x)

def f_real(xf):
    x = sympy.Symbol('x')
    return f_symbol().subs(x, xf).evalf()

xes = np.arange(x_start, x_end, step)
fs = [f_real(x) for x in xes]


def h_i(i):
    # return xes[i+1] - xes[i]
    return step


def mu_i(i):
    return h_i(i-1) / (h_i(i-1) + h_i(i))

def d_i(i):
    if i+1 > len(fs) - 1:
        return (6 / (h_i(i - 1) + h_i(i))) * ((fs[0] - fs[i]) / h_i(i) - (fs[i] - fs[i - 1]) / h_i(i - 1))
    return (6/(h_i(i-1) + h_i(i))) * ((fs[i+1] - fs[i])/h_i(i) - (fs[i]-fs[i-1])/h_i(i-1))


def lambda_i(i):
    return 1 - mu_i(i)


def t_i(i, x):
    return (x-xes[i])/h_i(i)


# returns spline function
def evaluate_cubic_splines():
    # step 1. Eval M - array of derivatives S'(x_i)
    # need to solve system M * m = D
    N = xes.size - 1
    system_size = xes.size
    M = np.zeros((system_size, system_size))
    D = np.zeros((system_size,))

    for i in range(2, N-1):
        M[i][i-1] = mu_i(i)
        M[i][i] = 2
        M[i][i+1] = lambda_i(i)

        D[i] = d_i(i)

    # add border clauses for type 3
    M[1][1] = 2
    M[1][2] = lambda_i(1)
    M[1][N] = mu_i(1)
    D[1] = d_i(1)
    #
    M[N-1][1] = lambda_i(N)
    M[N-1][N-1] = mu_i(N)
    M[N-1][N] = 2
    D[N] = d_i(N)
    #

    f_2_derivative = f_symbol().diff(sympy.abc.x).diff(sympy.abc.x)

    M[0][0] = 1
    D[0] = f_2_derivative.subs(sympy.abc.x, xes[0]).evalf()

    M[N][N] = 1
    D[N] = D[0]


    m_solved = np.linalg.solve(M, D)
    # step 2. given m_is - derivatives evaluate spline

    def S(x):
        i = np.searchsorted(xes, x) - 1
        if i == -1:
            i = 0

        if i == N:
            i -= 1
        t = t_i(i, x)
        return fs[i]*((1-t)**2)*(1+2*t) + fs[i+1] * (t**2) * (3-2*t) + m_solved[i]*h_i(i)*t*(1-t)**2 - m_solved[i+1]*h_i(i)*(t**2)*(1-t)

    return S



def main():
    plt.plot(xes, fs, 'bo', label="Табличная функция f(x)")
    high_density_xes = np.arange(x_start, x_end, 0.01)

    real_fs = [f_real(x) for x in high_density_xes]
    plt.plot(high_density_xes, real_fs, 'r,-', label='f(x)')

    S = evaluate_cubic_splines()

    interpolated_fs = [S(x) for x in high_density_xes]
    plt.plot(high_density_xes, interpolated_fs, 'g,-', label='S(x)')

    prepare_plot()
    plt.show()


if __name__ == '__main__':
    main()

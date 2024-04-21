import numpy as np
from scipy import optimize

D = 140  # depth
D0 = 80  # optimal depth
sigma1 = 1.4
sigma2 = 1.2
F = 1

def paramByStrat_de(Aj, Bj, Aa, Ba):
    """
    Работает не так, как надо, ищет параметры, в которых максимум при этой стратегии среди всех возможных параметров,
        а не параметры, при которых максимум в этой стратегии среди всех возможных стратегий,
            очевидно, то, что среди всех параметров максимум при этой стратегии находится в таких параметрах,
                не означает, что при этих параметрах среди всех стратегий максимум находится в этой стратегии
    """
    def fitByStrat(x):
        a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a = x

        M1 = sigma1 * (Aj + D)
        M2 = -sigma2 * (Aj + D + Bj/2)
        M3 = -2*(np.pi*Bj)**2
        M4 = -((Aj+D0)**2 + (Bj**2)/2)

        M5 = sigma1 * (Aa + D)
        M6 = -sigma2 * (Aa + D + Ba/2)
        M7 = -2*(np.pi*Ba)**2
        M8 = -((Aa+D0)**2 + (Ba**2)/2)

        p = a_j*M1 + b_j*M3 + d_j*M4
        r = a_a*M5 + b_a*M7 + d_a*M8
        q = -g_j*M2
        s = -g_a*M6

        if (4*r*p+(p+q*F-s*F)**2 < 0):
            fit = 1e9
        else:
            fit = -(-s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2))))

        return fit

    constraintBa = lambda x: -(x[1]*sigma1 - 2*x[7]*(Aa + D0))/(2*(4*np.pi**2 * x[3] + x[7]))
    nlc_AaBa = optimize.NonlinearConstraint(constraintBa, Ba - 1e-2, Ba + 1e-2)
    constraintBj = lambda x: -(x[0]*sigma1 - 2*x[6]*(Aj + D0))/(2*(4*np.pi**2 * x[2] + x[6]))
    nlc_AjBj = optimize.NonlinearConstraint(constraintBj, Bj - 1e-3, Bj + 1e-3)
    def constraint1(x):
        return (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F)
    nlc_BjBa = optimize.NonlinearConstraint(constraint1, 1 - 1e-6, 1 + 1e-6)

    return optimize.differential_evolution(fitByStrat, constraints=(nlc_AaBa, nlc_AjBj, nlc_BjBa),
    # bounds=[(0.013, 0.013), (0.042, 0.042), (0.0000062, 0.0000062), (0.0000047, 0.0000047), (0.00084, 0.00084), (0.037, 0.037), (0.00017, 0.00017), (0.0004, 0.0004)])
    # bounds=[(1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1)])
    # bounds=[(1e-3, 1e-1), (1e-3, 1e-1), (1e-6, 1e-4), (1e-6, 1e-4), (1e-5, 1e-3), (1e-3, 1e-1), (1e-5, 1e-3), (1e-4, 1e-2)])
    bounds=[(1e-3, 1e-1), (1e-3, 3e-1), (1e-5, 1e-2), (1e-5, 1e-3), (1e-5, 1e-1), (1e-3, 2e-1), (1e-5, 1e-2), (1e-5, 1e-2)])

def stratByParam_de(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a):
    def fitByParam(x):
        Aj, Bj, Aa, Ba = x

        M1 = sigma1 * (Aj + D)
        M2 = -sigma2 * (Aj + D + Bj/2)
        M3 = -2*(np.pi*Bj)**2
        M4 = -((Aj+D0)**2 + (Bj**2)/2)

        M5 = sigma1 * (Aa + D)
        M6 = -sigma2 * (Aa + D + Ba/2)
        M7 = -2*(np.pi*Ba)**2
        M8 = -((Aa+D0)**2 + (Ba**2)/2)

        p = a_j*M1 + b_j*M3 + d_j*M4
        r = a_a*M5 + b_a*M7 + d_a*M8
        q = -g_j*M2
        s = -g_a*M6

        if (4*r*p+(p+q*F-s*F)**2 < 0):
            fit = 1e9
        else:
            fit = -(-s*F-p-q*F+(np.sqrt((4*r*p+(p+q*F-s*F)**2))))

        return fit

    constraintBa = lambda x: -(a_a*sigma1 - 2*d_a*(x[2] + D0))/x[3]
    denominatorBa = (2*(4*np.pi**2 * b_a + d_a))
    nlc_AaBa = optimize.NonlinearConstraint(constraintBa, denominatorBa - 1e-2, denominatorBa + 1e-2)
    constraintBj = lambda x: -(a_j*sigma1 - 2*d_j*(x[0] + D0))/x[1]
    denominatorBj = (2*(4*np.pi**2 * b_j + d_j))
    nlc_AjBj = optimize.NonlinearConstraint(constraintBj, denominatorBj + 1e-3, denominatorBj + 1e-3)
    def constraint1(x):
        return (g_j*sigma2*F)/(2*(4*np.pi**2 * b_j + d_j)*x[1]) - (2*(4*np.pi**2 * b_a + d_a)*x[3])/(g_a*sigma2*F)
    nlc_BjBa = optimize.NonlinearConstraint(constraint1, 1 - 1e-6, 1 + 1e-6)

    lc_j1 = optimize.LinearConstraint([[1, 1, 0, 0]], -D, 0)
    lc_j2 = optimize.LinearConstraint([[1, -1, 0, 0]], -D, 0)
    lc_a1 = optimize.LinearConstraint([[0, 0, 1, 1]], -D, 0)
    lc_a2 = optimize.LinearConstraint([[0, 0, 1, -1]], -D, 0)
    return optimize.differential_evolution(fitByParam,
    bounds=[(-D, 0), (-D, 0), (-D, 0), (-D, 0)], constraints=(lc_j1, lc_j2, lc_a1, lc_a2))

def paramByStrat_fs(Aj, Bj, Aa, Ba):
    """Не работает вообще, т.к. эти формулы связи соот-ют стационарной точке для Aj, Bj, Aa, Ba, а не для параметров"""
    def func(_x):
        x = _x**2
        return [
        Bj + (x[0]*sigma1 - 2*x[6]*(Aj + D0))/(2*(4*np.pi**2 * x[2] + x[6])),
        Ba + (x[1]*sigma1 - 2*x[7]*(Aa + D0))/(2*(4*np.pi**2 * x[3] + x[7])),
        Bj + (x[0]*sigma1 - 2*x[6]*(Aj + D0))/(2*(4*np.pi**2 * x[2] + x[6]))
            -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F),
        Ba + (x[1]*sigma1 - 2*x[7]*(Aa + D0))/(2*(4*np.pi**2 * x[3] + x[7]))
            -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F),
        -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F),
        -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F),
        Bj + (x[0]*sigma1 - 2*x[6]*(Aj + D0))/(2*(4*np.pi**2 * x[2] + x[6]))
            -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F),
        Ba + (x[1]*sigma1 - 2*x[7]*(Aa + D0))/(2*(4*np.pi**2 * x[3] + x[7]))
            -1 + (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F)
        ]
    _root = optimize.fsolve(func, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print("err:", func(_root))
    print("sqrt(root):", _root)
    root = _root**2
    return root

if __name__ == "__main__":
    strat = stratByParam_de(0.013, 0.042, 0.0000063, 0.0000097, 0.0009, 0.036, 0.00017, 0.0002)
    print(strat.x, strat.fun)


    # param = paramByStrat_de(-34.37, -3.24, -82.82, -52.13)
    # print(param.x)

    # a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a = param.x
    # strat = stratByParam_de(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a)
    # print(strat.x, strat.fun)

    # strat = stratByParam_de(0.098, 0.29, 0.00037, 0.000022, 0.066, 0.19, 0.00043, 0.0032)
    # print(strat.x, strat.fun)


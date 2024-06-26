import numpy as np
from scipy import optimize

D = 140  # depth
D0 = 80  # optimal depth
sigma1 = 1.4
sigma2 = 1.2
F = 1


def stratByParam(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a):
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

def paramByStrat(Aj, Bj, Aa, Ba):
    def stratDeviation(x):
        a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a = x

        strat = stratByParam(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a)
        _Aj, _Bj, _Aa, _Ba = strat.x

        return np.abs(Aj-_Aj) + np.abs(Bj-_Bj) + np.abs(Aa-_Aa) + np.abs(Ba-_Ba)

    constraintBa = lambda x: -(x[1]*sigma1 - 2*x[7]*(Aa + D0))/(2*(4*np.pi**2 * x[3] + x[7]))
    nlc_AaBa = optimize.NonlinearConstraint(constraintBa, Ba - 1e-2, Ba + 1e-2)
    constraintBj = lambda x: -(x[0]*sigma1 - 2*x[6]*(Aj + D0))/(2*(4*np.pi**2 * x[2] + x[6]))
    nlc_AjBj = optimize.NonlinearConstraint(constraintBj, Bj - 1e-3, Bj + 1e-3)
    def constraint1(x):
        return (x[4]*sigma2*F)/(2*(4*np.pi**2 * x[2] + x[6])*Bj) - (2*(4*np.pi**2 * x[3] + x[7])*Ba)/(x[5]*sigma2*F)
    nlc_BjBa = optimize.NonlinearConstraint(constraint1, 1 - 1e-6, 1 + 1e-6)

    return optimize.differential_evolution(stratDeviation, constraints=(nlc_AaBa, nlc_AjBj, nlc_BjBa),
    bounds=[(1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1), (1e-6, 1)])

if __name__ == "__main__":
    #strat = stratByParam(0.00470259, 29089.2, 0.0000055, 0.0098, 1.73809, 36.282, 0.00005, 441)
    #strat = stratByParam(0.0000000470259, 0.290892, 0.000000000055, 0.000000098, 0.0000173809, 0.00036282, 0.0000000005, 0.00441)
    #strat = stratByParam(0.000000047,0.290892,0.000000000055,0.000000098,0.0000173809,0.00036282,0.0000000005,0.00441)
    strat = stratByParam(0.013, 0.042, 0.0000063, 0.0000097, 0.0009, 0.036, 0.00017, 0.0002)
    print(strat.x, strat.fun)


    # param = paramByStrat(-34.37, -3.24, -82.82, -52.13)
    # print(param.x)

    # a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a = param.x
    # strat = stratByParam(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a)
    # print(strat.x, strat.fun)

    # strat = stratByParam(0.098, 0.29, 0.00037, 0.000022, 0.066, 0.19, 0.00043, 0.0032)
    # print(strat.x, strat.fun)


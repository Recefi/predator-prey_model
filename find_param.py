import numpy as np
import pandas as pd
from scipy import optimize

import libs.gen_selection as gs

D = 140  # depth
D0 = 80  # optimal depth
sigma1 = 1.4
sigma2 = 1.2
env_params = (D, D0, sigma1, sigma2)
F = 1


def stratByParam_de(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    params = a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a
    
    def fitByStrat(x):
        Aj, Bj, Aa, Ba = x
        fit = - gs.calcFitByStrat_nan(Aj, Bj, Aa, Ba, params=params, env_params=env_params, F=F)
        if (np.isnan(fit)):
            fit = 1e9
        return fit

    # constraintBa = lambda x: -(a_a*sigma1 - 2*d_a*(x[2] + D0))/x[3]
    # denominatorBa = (2*(4*np.pi**2 * b_a + d_a))
    # nlc_AaBa = optimize.NonlinearConstraint(constraintBa, denominatorBa - 1e-2, denominatorBa + 1e-2)
    # constraintBj = lambda x: -(a_j*sigma1 - 2*d_j*(x[0] + D0))/x[1]
    # denominatorBj = (2*(4*np.pi**2 * b_j + d_j))
    # nlc_AjBj = optimize.NonlinearConstraint(constraintBj, denominatorBj + 1e-3, denominatorBj + 1e-3)
    # def constraint1(x):
    #     return (g_j*sigma2*F)/(2*(4*np.pi**2 * b_j + d_j)*x[1]) - (2*(4*np.pi**2 * b_a + d_a)*x[3])/(g_a*sigma2*F)
    # nlc_BjBa = optimize.NonlinearConstraint(constraint1, 1 - 1e-6, 1 + 1e-6)

    lc_j1 = optimize.LinearConstraint([[1, 1, 0, 0]], -D, 0)
    lc_j2 = optimize.LinearConstraint([[1, -1, 0, 0]], -D, 0)
    lc_a1 = optimize.LinearConstraint([[0, 0, 1, 1]], -D, 0)
    lc_a2 = optimize.LinearConstraint([[0, 0, 1, -1]], -D, 0)
    return optimize.differential_evolution(fitByStrat,
                        bounds=[(-D, 0), (-D, 0), (-D, 0), (-D, 0)], constraints=(lc_j1, lc_j2, lc_a1, lc_a2))

def stratByParam_shgo(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    params = a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

    def fitByStrat(x):
        Aj, Bj, Aa, Ba = x
        fit = - gs.calcFitByStrat_nan(Aj, Bj, Aa, Ba, params=params, env_params=env_params, F=F)
        if (np.isnan(fit)):
            fit = 1e9
        return fit

    lc_j1 = optimize.LinearConstraint([[1, 1, 0, 0]], -D, 0)
    lc_j2 = optimize.LinearConstraint([[1, -1, 0, 0]], -D, 0)
    lc_a1 = optimize.LinearConstraint([[0, 0, 1, 1]], -D, 0)
    lc_a2 = optimize.LinearConstraint([[0, 0, 1, -1]], -D, 0)
    return optimize.shgo(fitByStrat,
                        bounds=[(-D, 0), (-D, 0), (-D, 0), (-D, 0)], constraints=(lc_j1, lc_j2, lc_a1, lc_a2))

def checkOnGenlSel(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    params = a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

    Aj, Bj, Aa, Ba = gs.genGenlStratsMemOpt(Aj_left=-D, Aj_right=0, Aj_step=1, Bj_step=1,
                                            Aa_left=-D, Aa_right=0, Aa_step=1, Ba_step=1)
    Aj.extend([-D, -D+1, -D])
    Bj.extend([0, 0, 0])
    Aa.extend([-D, -D, -D+1])
    Ba.extend([0, 0, 0])
    fits_pp, fits_np, fits_pn, fits_nn = gs.calcGenlFitsMemOpt(Aj,Bj,Aa,Ba, params=params, env_params=env_params, F=F)
    df = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba,
                        'fits(++)': fits_pp, 'fits(-+)': fits_np, 'fits(+-)': fits_pn, 'fits(--)': fits_nn})
    df['fits(max)'] = df.loc[:, 'fits(++)':'fits(--)'].max(axis=1, skipna=True)
    df = df.sort_values('fits(max)', ascending=False, na_position='last')

    with pd.option_context('display.max_rows', None):
        print(df.head(1000))
        #print(df[((df['Aj'] == -34) | (df['Aj'] == -35)) & ((df['Aa'] == -83) | (df['Aa'] == -84))])

if __name__ == "__main__":
    strat = stratByParam_de(a_j=0.013, b_j=0.0000063, g_j=0.0009, d_j=0.00017,
                                a_a=0.042, b_a=0.0000097, g_a=0.036, d_a=0.0002)
    print(strat.x, strat.fun)
    # [-34.58, -3.29, -83.32, -51.57] ([-34.57773346, -3.29156444, -83.31583613, -51.57157067]) -1.0027498786

    strat = stratByParam_shgo(a_j=0.013, b_j=0.0000063, g_j=0.0009, d_j=0.00017,
                                a_a=0.042, b_a=0.0000097, g_a=0.036, d_a=0.0002)
    print(strat.x, strat.fun)
    # [-140, 0, -140, 0]  -2.073884


    checkOnGenlSel(a_j=0.013, b_j=0.0000063, g_j=0.0009, d_j=0.00017,
                                a_a=0.042, b_a=0.0000097, g_a=0.036, d_a=0.0002)

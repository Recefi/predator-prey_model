import numpy as np
import pandas as pd
import scipy.integrate as integrate
import time

import source.param as param
import source.csv_data as cd


def genStrats(n):
    """
    Генерирует n наборов стратегий (Aj,Bj,Aa,Ba) 
        в каждом по 4 стратегии отличающихся лишь знаком при (Bj,Ba)
            стратегии в наборе идут в порядке: (+,+), (-,+), (+,-), (-,-)
    """
    Aj = []
    Bj = []
    Aa = []
    Ba = []
    for i in range(n):
        a_j = np.random.random()*(-param.D)
        m_j = min(-a_j, a_j + param.D)
        b_j = np.random.uniform(0, m_j)
        a_a = np.random.random()*(-param.D)
        m_a = min(-a_a, a_a + param.D)
        b_a = np.random.uniform(0, m_a)

        Aj.append(a_j)
        Bj.append(b_j)
        Aa.append(a_a)
        Ba.append(b_a)

        Aj.append(a_j)
        Bj.append(-b_j)
        Aa.append(a_a)
        Ba.append(b_a)

        Aj.append(a_j)
        Bj.append(b_j)
        Aa.append(a_a)
        Ba.append(-b_a)

        Aj.append(a_j)
        Bj.append(-b_j)
        Aa.append(a_a)
        Ba.append(-b_a)

    stratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
    return stratData

def calcMps(stratData):
    Aj = stratData['Aj']
    Bj = stratData['Bj']
    Aa = stratData['Aa']
    Ba = stratData['Ba']

    Mps = []
    pqrs = []
    for i in Aj.index:
        M1 = param.sigma1 * (Aj[i] + param.D)
        M2 = -param.sigma2 * (Aj[i] + param.D + Bj[i]/2)
        M3 = -2*(np.pi*Bj[i])**2
        M4 = -((Aj[i]+param.D0)**2 + (Bj[i]**2)/2)

        M5 = param.sigma1 * (Aa[i] + param.D)
        M6 = -param.sigma2 * (Aa[i] + param.D + Ba[i]/2)
        M7 = -2*(np.pi*Ba[i])**2
        M8 = -((Aa[i]+param.D0)**2 + (Ba[i]**2)/2)

        res = [M1,M2,M3,M4,M5,M6,M7,M8]
        for m in range(8):
            for j in range(m,8):
                res.append(res[m+1]*res[j+1])
        Mps.append(res)

        p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
        r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
        q = -param.gamma_j*M2
        s = -param.gamma_a*M6
        pqrs.append([p, q, r, s])

    cols = []
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    mpData = pd.DataFrame(Mps, columns=cols, index=stratData.index)
    pqrsData = pd.DataFrame(pqrs, columns=["p", "q", "r", "s"], index=stratData.index)
    return mpData, pqrsData

def calcFitness(stratData, pqrsData):
    p = pqrsData['p']
    q = pqrsData['q']
    r = pqrsData['r']
    s = pqrsData['s']

    fitness = []
    indxs = []
    for i in pqrsData.index:
        if(4*r[i]*p[i]+(p[i]+q[i]-s[i])**2 >= 0):
            fit = -s[i]-p[i]-q[i]+(np.sqrt((4*r[i]*p[i]+(p[i]+q[i]-s[i])**2)))
            fitness.append(fit)
            indxs.append(i)
    fitData = pd.DataFrame(fitness, columns=['fit'], index=indxs)
    stratFitData = pd.concat([stratData.loc[fitData.index], fitData], axis=1)
    return stratFitData

def calcPopDynamics(pqrsData):
    start = time.time()
    n = len(pqrsData.index)
    p = pqrsData['p'].values
    q = pqrsData['q'].values
    r = pqrsData['r'].values
    s = pqrsData['s'].values

    def func(t, z):
        sumComp = 0
        sumDeath = 0
        for i in range(n):
            sumComp += (z[i] + z[i+n])
            sumDeath += (q[i]*z[i] + s[i]*z[i+n])

        F = z[2*n]
        result = []
        for i in range(n):
            result.append(-p[i]*z[i] - q[i]*z[i]*F + r[i]*z[i+n] - z[i]*sumComp)
        for i in range(n):
            result.append(p[i]*z[i] - s[i]*z[i+n]*F - z[i+n]*sumComp)
        result.append(sumDeath*F - F)

        return result
    
    z_0 = np.full(2*n, 0.0001)
    z_0 = np.append(z_0, 0.0001)

    pop = integrate.solve_ivp(func, t_span=[0, 500], y0=z_0, method='Radau', dense_output=True)
    t = np.linspace(0, 500, 10000)
    # dense_output=True need only for .sol(t)
    # .sol(t) is better than .y & .t !!!
    # max_step doesn't change .sol(t), it calculates in parallel when dense_output=True !!!

    indxs = []
    for i in range(n):
        indxs.append('z1_v'+str(pqrsData.index[i]))
    for i in range(n):
        indxs.append('z2_v'+str(pqrsData.index[i]))
    indxs.append('F')
    popData = pd.DataFrame(pop.sol(t), columns=t, index=indxs)
    end = time.time()
    print ("calcPopDynamics: ", end - start)
    return popData
    
def analyzePopDynamics(stratData, rawPopData, eps):
    n = len(stratData.index)
    t = len(rawPopData.columns)

    strats = []
    for i in range(n):
        strat = []
        for j in range(t):
            if (rawPopData.iloc[i,j] < eps and rawPopData.iloc[i+n,j] < eps):
                strat.append(rawPopData.columns[j])
                strat.append(rawPopData.iloc[i,j])
                strat.append(rawPopData.iloc[i+n,j])
                break
        if not strat:
            strat.append(rawPopData.columns[t-1])  #
            strat.append(rawPopData.iloc[i,t-1])
            strat.append(rawPopData.iloc[i+n,t-1])
        strats.append(strat)
    
    popData = pd.DataFrame(strats, columns=['t', 'z1', 'z2'], index=stratData.index)
    stratPopData = pd.concat([stratData, popData], axis=1)
    return stratPopData

def calcSelection(keyData, mpData):
    n = len(mpData.index)

    if (cd.getCallerName() == "fixed_pred"):
        fit = keyData['fit'].values
        def assignClass(i, j):
            if (fit[i] > fit[j]):
                elem = 1
            else:
                elem = -1
            return elem

    if (cd.getCallerName() == "dynam_pred"):
        t = keyData['t'].values
        z1 = keyData['z1'].values
        z2 = keyData['z2'].values
        def assignClass(i, j):
            if (t[i] == t[j]):
                if (abs(z1[i] + z2[i]) > abs(z1[j] + z2[j])):
                    elem = 1
                else:
                    elem = -1
            else:
                if (t[i] > t[j]):
                    elem = 1
                else:
                    elem = -1
            return elem

    sel = []
    for i in range(n):
        for j in range(n):  # выборка с обратными парами, иначе классы могут получаться сильно несбалансированными
            if (i == j): continue
            sel.append([assignClass(i, j)] + mpData.iloc[i].subtract(mpData.iloc[j]).to_list())
        # for j in (i+1, n):  # выборка без обратных пар, классы могут получаться сильно несбалансированными (проверить)
        #     sel.append([assignClass(i, j)] + mpData.iloc[i].subtract(mpData.iloc[j]).to_list())

    selData = pd.DataFrame(sel, columns=['class']+mpData.columns.to_list())
    return selData

def normSelection(selData):
    colMaxs = []
    for i in range(1, len(selData.columns)):
        max = selData.iloc[:, i].abs().max()
        selData.iloc[:, i]/=max
        colMaxs.append(max)
    return selData, colMaxs

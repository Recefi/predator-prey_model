import numpy as np
import pandas as pd
import scipy.integrate as integrate

import source.param as param


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
    for i in Aj.index:  # используем исходные индексы стратегий
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

def calcPopDynamics(pqrsData):
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

        result = []
        F = z[2*n]
        for i in range(n):
            result.append(-p[i]*z[i] - q[i]*z[i]*F + r[i]*z[i+n] - z[i]*sumComp)
        for i in range(n):
            result.append(p[i]*z[i] - s[i]*z[i+n]*F - z[i+n]*sumComp)
        result.append(sumDeath*F - F)

        return result
    
    z_0 = np.full(2*n, 0.0001)
    z_0 = np.append(z_0, 0.0001)
    t_span = np.array([0, 1000])

    pop = integrate.solve_ivp(func, t_span, z_0, method='RK45', dense_output=True)

    indxs = []
    for i in range(n):
        indxs.append('z1_v'+str(pqrsData.index[i]))
    for i in range(n):
        indxs.append('z2_v'+str(pqrsData.index[i]))
    indxs.append('F')
    popData = pd.DataFrame(pop.y, columns=pop.t, index=indxs)
    return popData
    
def analyzePopDynamics(stratIndxs, rawData, eps):
    n = len(stratIndxs)
    t = len(rawData.columns)

    strats = []
    for i in range(n):
        strat = []
        for j in range(t):
            if (rawData.iloc[i,j] < eps and rawData.iloc[i+n,j] < eps):
                strat.append(rawData.columns[j])
                strat.append(rawData.iloc[i,j])
                strat.append(rawData.iloc[i+n,j])
                break
        if not strat:
            strat.append(-1)
            strat.append(rawData.iloc[i,t-1])
            strat.append(rawData.iloc[i+n,t-1])
        strats.append(strat)
    
    data = pd.DataFrame(strats, columns=['t', 'z1', 'z2'], index=stratIndxs)
    return data

def calcSelection(mpData, popData):
    n = len(mpData.index)

    selection = []
    for i in range(n):
        for j in range(n):  # выборка с обратными парами, иначе классы могут получиться сильно несбалансированными
            if (i == j):
                continue
            elem = []
            if (popData.iloc[i, 0] > popData.iloc[j, 0]):
                elem.append(1)
            else:
                elem.append(0)
            elem.extend(mpData.iloc[i].subtract(mpData.iloc[j]).values)
            selection.append(elem)

    selData = pd.DataFrame(selection, columns=['class']+mpData.columns.to_list())
    return selData

def normSelection(selData):
    colMaxs = []
    for i in range(1, len(selData.columns)):
        max = selData.iloc[:, i].abs().max()
        selData.iloc[:, i]/=max
        colMaxs.append(max)
    return selData, colMaxs

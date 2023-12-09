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
    A_jun = []
    B_jun = []
    A_adult = []
    B_adult = []
    for i in range(n):
        a_j = np.random.random()*(-param.depth)
        m_j = min(-a_j, a_j+param.depth)
        b_j = np.random.uniform(0, m_j)
        a_a = np.random.random()*(-param.depth)
        m_a = min(-a_a, a_a+param.depth)
        b_a = np.random.uniform(0, m_a)

        A_jun.append(a_j)
        B_jun.append(b_j)
        A_adult.append(a_a)
        B_adult.append(b_a)

        A_jun.append(a_j)
        B_jun.append(-b_j)
        A_adult.append(a_a)
        B_adult.append(b_a)

        A_jun.append(a_j)
        B_jun.append(b_j)
        A_adult.append(a_a)
        B_adult.append(-b_a)

        A_jun.append(a_j)
        B_jun.append(-b_j)
        A_adult.append(a_a)
        B_adult.append(-b_a)

    return A_jun, B_jun, A_adult, B_adult

def calcMps(stratData):
    """
    Подсчет макропараметров
        Возвращает: Mps, OrigIndxs, pqrsData
            OrigIndxs[индекс Mps] = исходный индекс
                pqrsData в исходных индексах стратегий, а не в индексах массива Mps
    """
    A_jun = stratData['Aj']
    B_jun = stratData['Bj']
    A_adult = stratData['Aa']
    B_adult = stratData['Ba']

    k = 0
    Mps = []
    OrigIndxs = []
    pqrs = []
    for i in A_jun.index:  # используем исходные индексы стратегий
        res = []

        M1 = param.sigma1 * (A_jun[i] + param.depth)
        M2 = -param.sigma2 * (A_jun[i] + param.depth + B_jun[i]/2)
        M3 = -2*(np.pi*B_jun[i])**2
        M4 = -((A_jun[i]+param.optimal_depth)**2 + (B_jun[i]**2)/2)

        M5 = param.sigma1 * (A_adult[i] + param.depth)
        M6 = -param.sigma2 * (A_adult[i] + param.depth + B_adult[i]/2)
        M7 = -2*(np.pi*B_adult[i])**2
        M8 = -((A_adult[i]+param.optimal_depth)**2 + (B_adult[i]**2)/2)

        p = param.alpha_j*M1 + param.beta_j*M3 + param.delta_j*M4
        r = param.alpha_a*M5 + param.beta_a*M7 + param.delta_a*M8
        q = -param.gamma_j*M2
        s = -param.gamma_a*M6

        res = [M1,M2,M3,M4,M5,M6,M7,M8]

        for m in range(8):
            for j in range(m,8):
                res.append(res[m+1]*res[j+1])
        Mps.append(res)
        pqrs.append([p, q, r, s])

        OrigIndxs.append(i)
        k+=1
    print("strats:",k)

    pqrsData = pd.DataFrame(pqrs, columns=["p", "q", "r", "s"], index=OrigIndxs)
    return Mps, OrigIndxs, pqrsData

def calcPopData(pqrsData):
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
    
def analyzePopData(stratIndxs, rawData, eps):
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

def calcSelData1(mpData, popData):
    """Выборка без обратных пар"""
    n = len(mpData.index)

    selection = []
    for i in range(n):
        for j in range(i+1, n):
            elem = []
            if (popData.iloc[i, 0] > popData.iloc[j, 0]):
                elem.append(1)
            else:
                elem.append(0)
            elem.extend(mpData.iloc[i].subtract(mpData.iloc[j]).values)
            selection.append(elem)

    selData = pd.DataFrame(selection, columns=['class']+mpData.columns.to_list())
    return selData

def calcSelData2(mpData, popData):
    """Выборка с обратными парами (как раньше)"""
    n = len(mpData.index)

    selection = []
    for i in range(n):
        for j in range(n):
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

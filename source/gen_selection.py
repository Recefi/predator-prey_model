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

def calcStratsPop(pqrsData):
    strats = pqrsData.index
    n = len(strats)

    def func(t, z):
        p = pqrsData['p']
        q = pqrsData['q']
        r = pqrsData['r']
        s = pqrsData['s']

        sum1 = 0
        sum2 = 0
        for i in strats:
            sum1 += (z[i] + z[i+n])
            sum2 += (q[i]*z[i] + s[i]*z[i+n])

        result = []
        F = z[2*n]
        for i in strats:
            result.append(-p[i]*z[i] - q[i]*F*z[i] + r[i]*z[i+n] - z[i]*sum1)
        for i in strats:
            result.append(p[i]*z[i] - s[i]*F*z[i+n] - z[i+n]*sum1)
        result.append(F*sum2 - F)

        return result
    
    z_0 = np.full(2*n, 0.0001)
    z_0 = np.append(z_0, 0.0001)
    t_span = np.array([0, 100])

    pop = integrate.solve_ivp(func, t_span, z_0, method='RK45', dense_output=True)
    return pop
    
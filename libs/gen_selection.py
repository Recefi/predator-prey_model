import numpy as np
import pandas as pd
import scipy.integrate as integrate
import time
from scipy.optimize import fsolve
# import sympy
# import mpmath

import libs.param as param
import libs.utility as ut


def genStrats(n, distrA="uniform", by4=False):
    if (distrA == "uniform"):  # r12,r56:~0.96  # r34,r78:~-0.2  # =beta(1,1)
        a_j = np.random.uniform(0, -param.D, size=n)
        a_a = np.random.uniform(0, -param.D, size=n)
    if (distrA == "triangular"):   # r12,r56:~0.9  # r34,r78:~0.1  # распределения A и B примерно идентичны
        a_j = np.random.triangular(-param.D, -param.D/2, 0, size=n)
        a_a = np.random.triangular(-param.D, -param.D/2, 0, size=n)
    if (distrA == "beta"):  # по мере увеличения (a=b) A сужает диапозон и повышает пред.высоту гистограммы, B наоборот
        a_j = -param.D * np.random.beta(5, 5, size=n)
        a_a = -param.D * np.random.beta(5, 5, size=n)
        # при (4,4)  # r12,r56:~0.84  # r34,r78:~0.3
        # при (5,5)  # r12,r56:~0.8  # r34,r78:~0.5
        # при (6,6)  # r12,r56:~0.77  # r34,r78:~0.55
        # при (7,7)  # r12,r56:~0.75  # r34,r78:~0.65

    # ни в одном из случаев by4 ни на что особо не влияет,
    # максимум гистограммы норм.макропараметров немного лучше становятся и проекции за счет уменьшения разброса

    Aj = []
    Bj = []
    Aa = []
    Ba = []
    for i in range(n):
        m_j = min(-a_j[i], a_j[i] + param.D)
        m_a = min(-a_a[i], a_a[i] + param.D)

        if(by4):
            b_j = np.random.uniform(0, m_j)
            b_a = np.random.uniform(0, m_a)
            Aj.extend([a_j[i], a_j[i], a_j[i], a_j[i]])
            Bj.extend([b_j, -b_j, b_j, -b_j])
            Aa.extend([a_a[i], a_a[i], a_a[i], a_a[i]])
            Ba.extend([b_a, b_a, -b_a, -b_a])
        else:
            b_j = np.random.uniform(-m_j, m_j)
            b_a = np.random.uniform(-m_a, m_a)
            Aj.append(a_j[i])
            Bj.append(b_j)
            Aa.append(a_a[i])
            Ba.append(b_a)


    stratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
    return stratData

def calcMpData(stratData):
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
                res.append(res[m]*res[j])
        Mps.append(res)

    cols = []
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    mpData = pd.DataFrame(Mps, columns=cols, index=stratData.index)
    return mpData

def calcPqrsData(mpData, a_j=param.alpha_j, b_j=param.beta_j, g_j=param.gamma_j, d_j=param.delta_j,
                                a_a = param.alpha_a, b_a=param.beta_a, g_a=param.gamma_a, d_a=param.delta_a):
    pqrs = []
    for i in mpData.index:
        M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[i, 'M1':'M8']

        p = a_j*M1 + b_j*M3 + d_j*M4
        r = a_a*M5 + b_a*M7 + d_a*M8
        q = -g_j*M2
        s = -g_a*M6
        pqrs.append([p, q, r, s])

    pqrsData = pd.DataFrame(pqrs, columns=["p", "q", "r", "s"], index=mpData.index)
    return pqrsData

def calcStratFitData(stratData, pqrsData, F=1):
    p = pqrsData['p']
    q = pqrsData['q']
    r = pqrsData['r']
    s = pqrsData['s']

    fitness = []
    indxs = []
    for i in pqrsData.index:
        if(4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2 >= 0):
            fit = -s[i]*F-p[i]-q[i]*F+(np.sqrt((4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2)))
            fitness.append(fit)
            indxs.append(i)
    fitData = pd.DataFrame(fitness, columns=['fit'], index=indxs)
    stratFitData = pd.concat([stratData.loc[fitData.index], fitData], axis=1)
    return stratFitData

def calcPopDynamics(pqrsData, tMax=1000, tParts=10000, z0=0.01, F0=0.1):
    n = len(pqrsData.index)
    p = pqrsData['p'].values
    q = pqrsData['q'].values
    r = pqrsData['r'].values
    s = pqrsData['s'].values

    def func(t, z):
        sumComp = 0
        sumDeath = 0
        for i in range(n):
            if ((z[i] >= 0) & (z[i+n] >= 0)).all():
                sumComp += (z[i] + z[i+n])
                sumDeath += (q[i]*z[i] + s[i]*z[i+n])
            else:
                if (z[i] >= 0).all():
                    sumComp += z[i]
                    sumDeath += (q[i]*z[i])
                if (z[i+n] >= 0).all():
                    sumComp += z[i+n]
                    sumDeath += (s[i]*z[i+n])

        F = z[2*n]
        result = []
        for i in range(n):
            result.append(-p[i]*z[i] - q[i]*z[i]*F + r[i]*z[i+n] - z[i]*sumComp)
        for i in range(n):
            result.append(p[i]*z[i] - s[i]*z[i+n]*F - z[i+n]*sumComp)
        result.append(sumDeath*F - F)

        return result
    
    z_0 = np.full(2*n, z0)
    z_0 = np.append(z_0, F0)

    pop = integrate.solve_ivp(func, t_span=[0, tMax], y0=z_0, method='Radau', dense_output=True)
    t = np.linspace(0, tMax, tParts)
    # dense_output=True need only for .sol(t)
    # either .sol(t) or .y & .t

    indxs = []
    for i in range(n):
        indxs.append('z1_v'+str(pqrsData.index[i]))
    for i in range(n):
        indxs.append('z2_v'+str(pqrsData.index[i]))
    indxs.append('F')
    popData = pd.DataFrame(pop.sol(t), columns=t, index=indxs)
    return popData
    
def analyzePopDynamics(stratData, rawPopData, eps):
    n = len(stratData.index)
    t = len(rawPopData.columns)

    timeTicks = []
    indexes = []
    strats = []
    for i in range(n):
        strat = []
        for j in range(t):
            if (rawPopData.iloc[i,j] < 0 and rawPopData.iloc[i+n,j] < 0):
                if (np.isin(j, timeTicks)):
                    indexes.append(stratData.index[i])
                    strat.append(-1)
                else:
                    strat.append(rawPopData.columns[j])
                    strat.append(rawPopData.iloc[i,j])
                    strat.append(rawPopData.iloc[i+n,j])
                    strats.append(strat)
                    timeTicks.append(j)
                break
        if not strat:
            if (rawPopData.iloc[i,t-1] >= eps and rawPopData.iloc[i+n,t-1] >= eps):
                print(stratData.index[i], "not nullified")
                strat.append(rawPopData.columns[t-1])
                strat.append(rawPopData.iloc[i,t-1])
                strat.append(rawPopData.iloc[i+n,t-1])
                strats.append(strat)
            else:
                print(stratData.index[i], "not nullified, dropped")
                indexes.append(stratData.index[i])
    
    tmpStratData = stratData.drop(indexes)
    popData = pd.DataFrame(strats, columns=['t', 'z1', 'z2'], index=tmpStratData.index)
    stratPopData = pd.concat([tmpStratData, popData], axis=1)

    arrF = rawPopData.iloc[2*n].tail(10).round(4).values
    with np.printoptions(precision=4):
        print("F*: ", arrF)
    FLim = arrF[0]
    if (arrF == FLim).sum() != arrF.size:
        print("WARNING: F* haven't reached!!!")
    return stratPopData, FLim

def calcSelection(keyData, mpData):
    n = len(mpData.index)

    if (ut.getCallerName() == "static_pred"):
        fit = keyData['fit'].values
        def assignClass(i, j):
            if (fit[i] > fit[j]):
                elem = 1
            else:
                elem = -1
            return elem

    if (ut.getCallerName() == "dynamic_pred"):
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
        for j in range(i+1, n):  
            elemClass = assignClass(i, j)
            elemDiffs = mpData.iloc[i] - mpData.iloc[j]
            sel.append([elemClass] + elemDiffs.to_list())
            sel.append([-elemClass] + (-elemDiffs).to_list())
    # с обр.парами, чтобы получить выборку центрированную как по разностям макропараметров, так и по классам
    # за счет этого восст.гиперплоскость будет проходить примерно(с погр-тью до точн-ти класс-ра) ч-з центр координат
    #                                        и lam0 можно будет приравнять нулю (или сообщить класс-ру считать lam0 = 0)

    selData = pd.DataFrame(sel, columns=['class']+mpData.columns.to_list())
    return selData

def calcFLim(p, q, r, s, F0=0.1):  # в качестве стартовой оценки решения можно исп-ть нач.условие из задачи Коши
    def func1(F):
        return 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
                                                - ((-(p+q*F+s*F) + np.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
    def func(F):
        return np.abs(2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)))
                                                - ((-(p+q*F+s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F)
        # tmp1 = 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
        #                                 - ((-(p+q*F+s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
        # tmp2 = np.abs(tmp1)
        #print(tmp1, tmp2)
        #return tmp2

        # the modulus(euclidean norm) is the euclidean distance from 0 to the number, including complex number.
        # |a + bi| = sqrt(a^2 + b^2), the distance between the origin (0, 0) and the point (a, b) in the complex plane.
    
    root = fsolve(func, F0)
    #err1 = func1(root[0])
    err = func(root[0])
    # print(root[0])
    #print("err1:", err1)
    # print("err:", err)
    return root[0], err

def findFsols(p, q, r, s, left=-1000, right=1000, step=1, errEps = 1e-12, rndEps = 10):
    Fsols = []
    for F0 in range(left, right, step):
        F, err = calcFLim(p, q, r, s, F0)
        if (err < errEps):
            F = np.round(F, rndEps)
            if not Fsols:
                Fsols.append(F)
            elif (np.abs(F - Fsols) > 10**(-rndEps)).all():
                Fsols.append(F)
    return Fsols

def calcComplexFLim(p, q, r, s, F0=0.1):
    def tmpFunc(F):
        return 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
                                    - ((-(p+q*F+s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
    def complexFunc(F):
        re, im = F
        tmp = tmpFunc(complex(re, im))
        return [tmp.real, tmp.imag]
    
    root = fsolve(complexFunc, (F0, 1))
    F = complex(root[0], root[1])
    err = tmpFunc(F)
    # print(F)
    # print("err:", err)
    return F, err

def findComplexFsols(p, q, r, s, left=-1000, right=1000, step=1, errEps = 1e-15, reRndEps = 7, imRndEps = 6):
    Fsols = []
    for F0 in range(left, right, step):
        F, err = calcComplexFLim(p, q, r, s, F0)
        if (err.real < errEps):
            F = np.round(F, reRndEps)
            if not Fsols:
                Fsols.append(F)
            elif((np.abs(F.real-np.real(Fsols)) > 10**(-reRndEps)).all()
            or (np.abs(F.imag-np.imag(Fsols)) > 10**(-imRndEps)).all()):
                Fsols.append(F)
    return Fsols

# def calcFLimSympy(p, q, r, s, F0=0.1):
#     F = sympy.symbols('F')
#     equation = sympy.Eq(2*r*p / (2*p*s + q*(-(p+q*F-s*F) + sympy.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
#                                     - ((-(p+q*F+s*F) + sympy.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F, 0)
#     sols = sympy.nsolve(equation, F, F0)
    
#     # def func(F):
#     #     return 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + sympy.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
#     #                                             - ((-(p+q*F+s*F) + sympy.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
#     # sols = mpmath.findroot(func, 0)
#     print("sols:", sols)

def calcZLim(p, q, r, s, F):
    z2 = 2*p/(2*p*s + q*s*F - q*(p+q*F) + q*np.sqrt(4*r*p+(p+q*F-s*F)**2))
    z1 = (s*F - (p+q*F) + np.sqrt(4*r*p+(p+q*F-s*F)**2))/(2*p) * z2
    return z1, z2

def checkFLim(p, q, r, s, F, z1, z2):
    a11, a12, a13 = (-p - q*F - z1 - z2, -z1 + r, -q*z1)
    a21, a22, a23 = (p-z2, -s*F - z1 - z2, -s*z2)
    a31, a32, a33 = (q*F, s*F, q*z1 + s*z2 - 1)

    #print("!!!test!!!", -q*z1, a13)

    pows = [1, -(a11 + a22 + a33), (a11*a22 + a11*a33 + a22*a33 - a31*a13 - a32*a23 - a12*a21),
            -(a11*a22*a33 + a21*a32*a13 + a12*a23*a31 - a31*a13*a22 - a32*a23*a11 - a12*a21*a33)]
    L = np.roots(pows)
    def err(L):
        return (a11-L)*(a22-L)*(a33-L) + a21*a32*a13 + a12*a23*a31 - a31*a13*(a22-L) - a23*a32*(a11-L) - a12*a21*(a33-L)
    errs = [err(L[0]), err(L[1]), err(L[2])]

    print("roots: ", L)
    print("errs: ", errs)
    return L, errs

def checkFsols(p, q, r, s, Fsols):
    Flams = []
    lamsErrs = []
    for i in range(len(Fsols)):
        z1, z2 = calcZLim(p, q, r, s, Fsols[i])
        roots, errs = checkFLim(p, q, r, s, Fsols[i], z1, z2)
        if (roots.real > 0).any():
            Flams.append('+')
        else:
            Flams.append('-')
        lamsErrs.append(errs)
    return Flams, lamsErrs

def checkFsolsOnSel(stratData, pqrsData):
    res = []
    for i in pqrsData.index:
        resStr = []
        p, q, r, s = pqrsData.loc[i, ['p', 'q', 'r', 's']]
        Fsols = findFsols(p, q, r, s)
        Flams, lamsErrs = checkFsols(p, q, r, s, Fsols)
        #resStr.append(Fsols)
        resStr.append(Flams)
        resStr.append(lamsErrs)
        res.append(resStr)
    #FsolsData = stratData.copy()
    #FsolsData.loc[:, 'Fsols'] = FsolsList
    #FsolsData.loc[:, 'Flams'] = FlamsList
    #FsolsData.loc[:, 'errs'] = errsList
    FsolsData = pd.DataFrame(res, columns=['Flams', 'lamsErrs'], index=pqrsData.index)
    return FsolsData

def checkComplexFsolsOnSel(stratData, pqrsData):
    #FsolsList = []
    FlamsList = []
    lamsErrsList = []
    for i in pqrsData.index:
        resStr = []
        p, q, r, s = pqrsData.loc[i, ['p', 'q', 'r', 's']]
        Fsols = findComplexFsols(p, q, r, s)
        Flams, lamsErrs = checkFsols(p, q, r, s, Fsols)
        #FsolsList.append(Fsols)
        FlamsList.append(Flams)
        lamsErrsList.append(lamsErrs)
    complexFsolsData = stratData.copy()
    for j in range(len(max(FlamsList, key=len))):
        complexFsolsData.loc[:, 'Fsol_'+str(j)] = FlamsList[:, j]
    for j in range(len(max(lamsErrsList, key=len))):
        complexFsolsData.loc[:, 'lamsErrs_'+str(j)] = lamsErrsList[:, j]
    return complexFsolsData

def fitBySel(stratData, pqrsData):
    p = pqrsData['p']
    q = pqrsData['q']
    r = pqrsData['r']
    s = pqrsData['s']

    indxs = []
    mins = []
    counts = []
    for j in pqrsData.index:
        print(j)
        F, err = calcFLim(p[j], q[j], r[j], s[j], F0=0.1)
        next = 4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2 < 0
        if (not next):
            z1, z2 = calcZLim(p[j], q[j], r[j], s[j], F)
            roots, errs = checkFLim(p[j], q[j], r[j], s[j], F, z1, z2)
            next = (roots.real > 0).any()
        if next:
            F, err = calcFLim(p[j], q[j], r[j], s[j], F0=-100000)
            next = 4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2 < 0
            if (not next):
                z1, z2 = calcZLim(p[j], q[j], r[j], s[j], F)
                roots, errs = checkFLim(p[j], q[j], r[j], s[j], F, z1, z2)
                next = (roots.real > 0).any()
            if next:
                F, err = calcFLim(p[j], q[j], r[j], s[j], F0=100000)
                next = 4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2 < 0
                if (not next):
                    z1, z2 = calcZLim(p[j], q[j], r[j], s[j], F)
                    roots, errs = checkFLim(p[j], q[j], r[j], s[j], F, z1, z2)
                    next = (roots.real > 0).any()
                if next:
                    print("!!! WARNING: both steady states are not suitable !!!", j)
                    continue

        min = 1
        count = 0
        for i in pqrsData.index:
            if(4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2 >= 0):
                count += 1
                fit = -s[j]*F-p[j]-q[j]*F+(np.sqrt((4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2))) \
                    - (-s[i]*F-p[i]-q[i]*F+(np.sqrt((4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2))))
                if (fit < min):
                    min = fit
        indxs.append(j)
        mins.append(min)
        counts.append(count)
    
    stratMinsData = stratData.loc[indxs]
    stratMinsData.loc[:, 'min'] = mins
    stratMinsData.loc[:, 'count'] = counts
    idOptStrat = stratMinsData['min'].idxmax()
    return stratMinsData, idOptStrat

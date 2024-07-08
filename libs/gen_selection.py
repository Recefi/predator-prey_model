import numpy as np
import pandas as pd
import scipy as sp
import time
import tqdm
from numba import jit, njit, prange
from joblib import Parallel, delayed
import itertools

import libs.param as param
import libs.utility as ut


def genStrats(n, distrA="uniform", by4=False, ab=5):
    if (distrA == "uniform"):  # r12,r56:~0.96  # r34,r78:~-0.2  # =beta(1,1)
        a_j = np.random.uniform(0, -param.D, size=n)
        a_a = np.random.uniform(0, -param.D, size=n)
    if (distrA == "triangular"):   # r12,r56:~0.9  # r34,r78:~0.1  # распределения A и B примерно идентичны
        a_j = np.random.triangular(-param.D, -param.D/2, 0, size=n)
        a_a = np.random.triangular(-param.D, -param.D/2, 0, size=n)
    if (distrA == "beta"):  # по мере увеличения (a=b) A сужает диапозон и повышает пред.высоту гистограммы, B наоборот
        a_j = -param.D * np.random.beta(ab, ab, size=n)
        a_a = -param.D * np.random.beta(ab, ab, size=n)
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
    M1, M2, M3, M4, M5, M6, M7, M8 = (mpData[col].values for col in mpData[['M1','M2','M3','M4','M5','M6','M7','M8']])
    pqrs = []
    for i in range(len(M1)):
        p = a_j*M1[i] + b_j*M3[i] + d_j*M4[i]
        r = a_a*M5[i] + b_a*M7[i] + d_a*M8[i]
        q = -g_j*M2[i]
        s = -g_a*M6[i]
        pqrs.append([p, q, r, s])

    pqrsData = pd.DataFrame(pqrs, columns=["p", "q", "r", "s"], index=mpData.index)
    return pqrsData

def calcLinsum(mpMatr, lams):
    """lams = [lam1,...]"""
    fitness = []
    for i in range(len(mpMatr)):
        fit = 0
        for j in range(len(mpMatr[i])):
            fit += lams[j]*mpMatr[i, j]
        fitness.append(fit)
    return fitness

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

def calcStratFitData_linsum(stratData, mpData, coefData, idx=-1):
    mpMatr = mpData.values
    lams = coefData.iloc[idx].values
    fitness = calcLinsum(mpMatr, lams)

    fitData = pd.DataFrame(fitness, columns=['fit'], index=mpData.index)
    stratFitData = pd.concat([stratData.loc[fitData.index], fitData], axis=1)
    return stratFitData

@njit
def integrateIter(t, z, n, p, q, r, s):
    sumComp = 0
    sumDeath = 0
    for i in range(n):
        if (z[i] > 0 and z[i+n] > 0):  # otherwise: Radau,BDF,LSODA time is more(x1.1), number of strats is indifferent,
                                          # (only!)Radau error is less, but still more than BDF error and less LSODA.
            sumComp += (z[i] + z[i+n])
            sumDeath += (q[i]*z[i] + s[i]*z[i+n])
        else:
            if (z[i] > 0):
                sumComp += z[i]
                sumDeath += (q[i]*z[i])
            if (z[i+n] > 0):
                sumComp += z[i+n]
                sumDeath += (s[i]*z[i+n])
    F = z[2*n]
    result = np.empty(2*n + 1)
    for i in range(n):
        result[i] = -p[i]*z[i] - q[i]*z[i]*F + r[i]*z[i+n] - z[i]*sumComp
        result[i+n] = p[i]*z[i] - s[i]*z[i+n]*F - z[i+n]*sumComp
    result[2*n] = sumDeath*F - F
    return result

def calcPopDynamics(pqrsData, tMax=1000, tParts=10001, z0=0.01, F0=0.1, _method='BDF'):
    n = len(pqrsData.index)
    p = pqrsData['p'].values
    q = pqrsData['q'].values
    r = pqrsData['r'].values
    s = pqrsData['s'].values

    z_0 = np.full(2*n, z0)
    z_0 = np.append(z_0, F0)

    t = np.linspace(0, tMax, tParts)
    pop = sp.integrate.solve_ivp(integrateIter,args=(n, p, q, r, s), t_span=[0, tMax], t_eval=t, y0=z_0, method=_method)

    indxs = []
    for i in range(n):
        indxs.append('z1_v'+str(pqrsData.index[i]))
    for i in range(n):
        indxs.append('z2_v'+str(pqrsData.index[i]))
    indxs.append('F')
    popData = pd.DataFrame(pop.y, columns=pop.t, index=indxs)
    return popData

def analyzePopDynamics(stratData, rawPopData, eps):
    n = len(stratData.index)
    t = len(rawPopData.columns)
    rawPopMatr = rawPopData.values  # 38s ---> 0.44s !!!

    timeTicks = []
    indexes = []
    strats = []
    for i in range(n):
        strat = []
        for j in range(t):
            if (rawPopMatr[i,j] < 0 and rawPopMatr[i+n,j] < 0):
                if (np.isin(j, timeTicks)):  # drop strat
                    indexes.append(stratData.index[i])
                    strat.append(-1)
                else:
                    strat.append(rawPopData.columns[j])
                    strat.append(rawPopMatr[i,j])
                    strat.append(rawPopMatr[i+n,j])
                    strats.append(strat)
                    timeTicks.append(j)
                break
        if not strat:
            if (rawPopMatr[i,t-1] >= eps and rawPopMatr[i+n,t-1] >= eps):
                print(stratData.index[i], "not nullified")
                strat.append(rawPopData.columns[t-1])
                strat.append(rawPopMatr[i,t-1])
                strat.append(rawPopMatr[i+n,t-1])
                strats.append(strat)
            else:  # drop strat
                print(stratData.index[i], "not nullified, dropped")
                indexes.append(stratData.index[i])

    tmpStratData = stratData.drop(indexes)
    popData = pd.DataFrame(strats, columns=['t', 'z1', 'z2'], index=tmpStratData.index)
    stratPopData = pd.concat([tmpStratData, popData], axis=1)

    arrF = rawPopData.iloc[2*n].tail(10).round(10).values
    with np.printoptions(precision=10):
        print("F*: ", arrF)
    FLim = arrF[0]
    if (arrF == FLim).sum() != arrF.size:
        print("WARNING: F* haven't reached!!!")
    return stratPopData, FLim

def calcSelection(keyData, mpData, callerName=""):
    n = len(mpData.index)
    mpMatr = mpData.values  # 17.84s ---> 2.27s !!!

    if not callerName:
        callerName = ut.getCallerName()
    if (callerName == "static_pred"):
        fit = keyData['fit'].values
        def assignClass(i, j):
            if (fit[i] > fit[j]):
                elem = 1
            else:
                elem = -1
            return elem
    if (callerName == "dynamic_pred"):
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
            elemDiffs = mpMatr[i] - mpMatr[j]
            sel.append([elemClass] + elemDiffs.tolist())
            sel.append([-elemClass] + (-elemDiffs).tolist())
    # с обр.парами, чтобы получить выборку центрированную как по разностям макропараметров, так и по классам
    # за счет этого восст.гиперплоскость будет проходить примерно(с погр-тью до точн-ти класс-ра) ч-з центр координат
    #                                        и lam0 можно будет приравнять нулю (или сообщить класс-ру считать lam0 = 0)

    selData = pd.DataFrame(sel, columns=['class']+mpData.columns.to_list())
    return selData

def genGenlStrats(a_j=param.alpha_j, b_j=param.beta_j, g_j=param.gamma_j, d_j=param.delta_j,
                    a_a = param.alpha_a, b_a=param.beta_a, g_a=param.gamma_a, d_a=param.delta_a):
    Aj = []
    Bj = []
    Aa = []
    Ba = []
    for i in tqdm.tqdm(range(-1, -param.D*10, -1)):
        A_j = i*0.1
        m_j = min(-A_j, A_j + param.D)
        B_j = -(a_j*param.sigma1 - 2*d_j*(A_j + param.D0))/(2*(4*np.pi**2 * b_j + d_j))
        if (-m_j <= B_j and B_j <= m_j):
            for j in range(-1, -param.D*10, -1):
                A_a = j*0.1
                m_a = min(-A_a, A_a + param.D)
                B_a = -(a_a*param.sigma1 - 2*d_a*(A_a + param.D0))/(2*(4*np.pi**2 * b_a + d_a))
                if (-m_a <= B_a and B_a <= m_a):
                    Aj.append(A_j)
                    Bj.append(B_j)
                    Aa.append(A_a)
                    Ba.append(B_a)
    return Aj, Bj, Aa, Ba

def genGenlStratsAll(Aj_left=-param.D+1, Aj_right=0, Aj_step=2, Bj_step=1,
                        Aa_left=-param.D+1, Aa_right=0, Aa_step=2, Ba_step=1):
    Aj = []
    Bj = []
    Aa = []
    Ba = []
    for A_j in tqdm.tqdm(range(Aj_left, Aj_right, Aj_step)):
        m_j = min(-A_j, A_j + param.D)
        for B_j in itertools.chain(range(-m_j, 0, Bj_step), range(1, m_j+1, Bj_step)):
            for A_a in range(Aa_left, Aa_right, Aa_step):
                m_a = min(-A_a, A_a + param.D)
                for B_a in itertools.chain(range(-m_a, 0, Ba_step), range(1, m_a+1, Ba_step)):
                    Aj.append(A_j)
                    Bj.append(B_j)
                    Aa.append(A_a)
                    Ba.append(B_a)
    return Aj, Bj, Aa, Ba

def calcGenlPqrsData(Aj, Bj, Aa, Ba, a_j=param.alpha_j, b_j=param.beta_j, g_j=param.gamma_j, d_j=param.delta_j,
                                a_a = param.alpha_a, b_a=param.beta_a, g_a=param.gamma_a, d_a=param.delta_a):
    p = np.empty(len(Aj))
    q = np.empty(len(Aj))
    r = np.empty(len(Aj))
    s = np.empty(len(Aj))
    for i in tqdm.tqdm(range(len(Aj))):
        M1 = param.sigma1 * (Aj[i] + param.D)
        M2 = -param.sigma2 * (Aj[i] + param.D + Bj[i]/2)
        M3 = -2*(np.pi*Bj[i])**2
        M4 = -((Aj[i]+param.D0)**2 + (Bj[i]**2)/2)

        M5 = param.sigma1 * (Aa[i] + param.D)
        M6 = -param.sigma2 * (Aa[i] + param.D + Ba[i]/2)
        M7 = -2*(np.pi*Ba[i])**2
        M8 = -((Aa[i]+param.D0)**2 + (Ba[i]**2)/2)

        p[i] = a_j*M1 + b_j*M3 + d_j*M4
        r[i] = a_a*M5 + b_a*M7 + d_a*M8
        q[i] = -g_j*M2
        s[i] = -g_a*M6

    return p, q, r, s

def calcFLim(p,q,r,s, F0=0.1, abs=True):  # в качестве стартовой оценки решения можно исп-ть нач.условие из задачи Коши
    def func1(F):
        return 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
                                                - ((-(p+q*F+s*F) + np.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
    def func(F):
        return np.abs(2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)))
                                                - ((-(p+q*F+s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F)
        # the modulus(euclidean norm) is the euclidean distance from 0 to the number, including complex number.
        # |a + bi| = sqrt(a^2 + b^2), the distance between the origin (0, 0) and the point (a, b) in the complex plane.

    root = sp.optimize.fsolve(func, F0) if abs else sp.optimize.fsolve(func1, F0)
    #err1 = func1(root[0])
    err = func(root[0])
    # print(root[0])
    #print("err1:", err1)
    # print("err:", err)
    return root[0], err

def calcZLim(p, q, r, s, F):
    z2 = 2*p/(2*p*s + q*s*F - q*(p+q*F) + q*np.sqrt(4*r*p+(p+q*F-s*F)**2))
    z1 = (s*F - (p+q*F) + np.sqrt(4*r*p+(p+q*F-s*F)**2))/(2*p) * z2
    return z1, z2

def calcZLim_auto(p, q, r, s, F):
    def func(z):
        res = []
        res.append(z[0]*(-p-q*F-z[0]-z[1]) + r*z[1])
        res.append(z[1]*(-s*F-z[0]-z[1]) + p*z[0])
        res.append(F*(q*z[0] + s*z[1] - 1))
        return res

    roots = sp.optimize.fsolve(func, [0.1, 0.1, 0])
    # errs = func(roots)
    # print(errs)
    return roots[0], roots[1]

def chkFLim(p, q, r, s, F, z1, z2):
    a11, a12, a13 = (-p - q*F - z1 - z2, -z1 + r, -q*z1)
    a21, a22, a23 = (p-z2, -s*F - z1 - z2, -s*z2)
    a31, a32, a33 = (q*F, s*F, q*z1 + s*z2 - 1)

    pows = [1, -(a11 + a22 + a33), (a11*a22 + a11*a33 + a22*a33 - a31*a13 - a32*a23 - a12*a21),
            -(a11*a22*a33 + a21*a32*a13 + a12*a23*a31 - a31*a13*a22 - a32*a23*a11 - a12*a21*a33)]
    L = np.roots(pows)
    def err(L):
        return (a11-L)*(a22-L)*(a33-L) + a21*a32*a13 + a12*a23*a31 - a31*a13*(a22-L) - a23*a32*(a11-L) - a12*a21*(a33-L)
    errs = [err(L[0]), err(L[1]), err(L[2])]

    # print("roots: ", L)
    # print("errs: ", errs)
    return L, errs

def chkFLim_auto(p, q, r, s, F, z1, z2):
    A = np.array([
    [-p - q*F - z1 - z2, -z1 + r, -q*z1],
    [p-z2, -s*F - z1 - z2, -s*z2],
    [q*F, s*F, q*z1 + s*z2 - 1]
    ])
    L = np.linalg.eig(A).eigenvalues

    # print("roots: ", L)
    return L

def findF(p, q, r, s, j):
    p, q, r, s = p[j], q[j], r[j], s[j]
    F, err = calcFLim(p, q, r, s, F0=0.1)
    # next = 4*r*p+(p+q*F-s*F)**2 < 0
    # if (not next):
    #     z1, z2 = calcZLim(p, q, r, s, F)
    #     roots, errs = chkFLim(p, q, r, s, F, z1, z2)
    #     next = (roots.real > 0).any()
    # if next:
    #     F, err = calcFLim(p, q, r, s, F0=-100)
    #     next = 4*r*p+(p+q*F-s*F)**2 < 0
    #     if (not next):
    #         z1, z2 = calcZLim(p, q, r, s, F)
    #         roots, errs = chkFLim(p, q, r, s, F, z1, z2)
    #         next = (roots.real > 0).any()
    #     if next:
    #         F, err = calcFLim(p, q, r, s, F0=100)
    if (4*r*p+(p+q*F-s*F)**2 < 0):
        #return [1, j]  # use this strat as i
        return [0]
    else:
        z1, z2 = calcZLim(p, q, r, s, F)
        roots, errs = chkFLim(p, q, r, s, F, z1, z2)
        if (roots.real > 0).any():
            return [0]
    return [2, j, F]

@njit(parallel=True)
def findMins(p, q, r, s, Fs, Fsj, Fsi):
    mins = np.empty(len(Fs))
    for _j in prange(len(Fs)):
        j = Fsj[_j]
        F = Fs[_j]

        fitj = -s[j]*F-p[j]-q[j]*F+(np.sqrt((4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2)))
        min = 1
        for _i in range(len(Fsi)):
            i = Fsi[_i]
            tmp = 4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2
            if(tmp >= 0):
                fit = fitj - (-s[i]*F-p[i]-q[i]*F+np.sqrt(tmp))
                if (fit < min):
                    min = fit
        mins[_j] = min
    return mins

def genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s, index=None):
    res = Parallel(n_jobs=-1)(delayed(findF)(p, q, r, s, j) for j in tqdm.tqdm(range(len(p))))
    start = time.time()
    Fs = [item[2] for item in res if (item[0] == 2)]
    Fsj = [item[1] for item in res if (item[0] == 2)]
    Fsi = [item[1] for item in res if (item[0])]
    print ("analyze res list: ", time.time() - start)

    start = time.time()
    mins = findMins(p, q, r, s, Fs, Fsj, Fsi)
    print ("calc genl mins time: ", time.time() - start)

    _Aj = [Aj[j] for j in Fsj]
    _Bj = [Bj[j] for j in Fsj]
    _Aa = [Aa[j] for j in Fsj]
    _Ba = [Ba[j] for j in Fsj]
    if index is not None:
        Fsj = [index[j] for j in Fsj]
    stratMinsData = pd.DataFrame({'Aj': _Aj, 'Bj': _Bj, 'Aa': _Aa, 'Ba': _Ba, 'min': mins}, index=Fsj)
    idOptStrat = stratMinsData['min'].idxmax()
    return stratMinsData, idOptStrat

def fitMaxMin(stratData, pqrsData):
    if (stratData.index != pqrsData.index).all():
        print("WARNING: stratData.index != pqrsData.index")

    Aj = stratData['Aj'].tolist()
    Bj = stratData['Bj'].tolist()
    Aa = stratData['Aa'].tolist()
    Ba = stratData['Ba'].tolist()

    p = pqrsData['p'].values
    q = pqrsData['q'].values
    r = pqrsData['r'].values
    s = pqrsData['s'].values

    return genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s, index=pqrsData.index)

def calcFLim_2(p1,q1,r1,s1, p2,q2,r2,s2, F0=0.1, abs=True):
    def func1(F):
        return -s1*F-p1-q1*F+np.sqrt(4*r1*p1+(p1+q1*F-s1*F)**2) - (-s2*F-p2-q2*F+np.sqrt(4*r2*p2+(p2+q2*F-s2*F)**2))

    def func(F):
        return np.abs(-s1*F-p1-q1*F+np.emath.sqrt(4*r1*p1+(p1+q1*F-s1*F)**2)
                    -(-s2*F-p2-q2*F+np.emath.sqrt(4*r2*p2+(p2+q2*F-s2*F)**2)))
        # the modulus(euclidean norm) is the euclidean distance from 0 to the number, including complex number.
        # |a + bi| = sqrt(a^2 + b^2), the distance between the origin (0, 0) and the point (a, b) in the complex plane.

    root = sp.optimize.fsolve(func, F0) if abs else sp.optimize.fsolve(func1, F0)
    # err1 = func1(root[0])
    err = func(root[0])
    # print(root[0])
    # print("err1:", err1)
    # print("err:", err)
    return root[0], err

def calcZLim_2(p1, q1, r1, s1, p2, q2, r2, s2, F):
    def func(z):
        res = []
        res.append(z[0]*(-p1-q1*F-z[0]-z[1]-z[2]-z[3]) + r1*z[1])
        res.append(z[1]*(-s1*F-z[0]-z[1]-z[2]-z[3]) + p1*z[0])
        res.append(z[2]*(-p2-q2*F-z[0]-z[1]-z[2]-z[3]) + r2*z[3])
        res.append(z[3]*(-s2*F-z[0]-z[1]-z[2]-z[3]) + p2*z[2])
        res.append(F*(q1*z[0] + s1*z[1] + q2*z[2] + s2*z[3] - 1))
        return res

    roots = sp.optimize.fsolve(func, [0.1, 0.1, 0.1, 0.1, 0])
    # errs = func(roots)
    # print(errs)
    return roots[0], roots[1], roots[2], roots[3]

def chkFLim_2(p1, q1, r1, s1, p2, q2, r2, s2, F, z11, z12, z21, z22):
    A = np.array([
    [-p1-q1*F-z11-z12-z21-z22,  -z11+r1,                -z11,                       -z11,                   -q1*z11],
    [p1-z12,                    -s1*F-z11-z12-z21-z22,  -z12,                       -z12,                   -s1*z12],
    [-z21,                      -z21,                   -p2-q2*F-z11-z12-z21-z22,   -z21+r2,                -q2*z21],
    [-z22,                      -z22,                   p2-z22,                     -s2*F-z11-z12-z21-z22,  -s2*z22],
    [q1*F,                      s1*F,                   q2*F,                       s2*F, q1*z11+s1*z12+q2*z21+s2*z22-1]
    ])
    L = np.linalg.eig(A).eigenvalues
    #L = sp.linalg.eigvals(A)

    # print("roots: ", L)
    return L

def findF_2(p, q, r, s, j, w):
    if (j == w):
        return [0]
    _p, _q, _r, _s = p[j], q[j], r[j], s[j]
    __p, __q, __r, __s = p[w], q[w], r[w], s[w]
    F, err = calcFLim_2(_p, _q, _r, _s, __p, __q, __r, __s, F0=0.1)
    if (4*_r*_p+(_p+_q*F-_s*F)**2 < 0 or 4*__r*__p+(__p+__q*F-__s*F)**2 < 0):
        #return [1, [j,w]]  # use this strat as i
        return [0]
    return [2, [j,w], F]

#@njit(parallel=True)
def findMins_2(p, q, r, s, Fs, Fsj, Fsi):
    mins = np.empty(len(Fs))
    for _j in prange(len(Fs)):
        j = Fsj[_j]
        F = Fs[_j]

        fitj = -s[j]*F-p[j]-q[j]*F+(np.sqrt((4*r[j]*p[j]+(p[j]+q[j]*F-s[j]*F)**2)))
        min = 1
        for _i in range(len(Fsi)):
            i = Fsi[_i]
            tmp = 4*r[i]*p[i]+(p[i]+q[i]*F-s[i]*F)**2
            if(tmp >= 0):
                fit = fitj - (-s[i]*F-p[i]-q[i]*F+np.sqrt(tmp))
                print('j:',j,' F:',F,' fit:',fit,' fitj:',fitj,' fiti:',-s[i]*F-p[i]-q[i]*F+np.sqrt(tmp),' i:',i,sep='')
                if (fit < min):
                    min = fit
        mins[_j] = min
    return mins

def genlFitMaxMin_2(Aj, Bj, Aa, Ba, p, q, r, s, index=None):
    res = []
    for j in tqdm.tqdm(range(len(p))):
        res.extend(Parallel(n_jobs=-1)(delayed(findF_2)(p, q, r, s, j, w) for w in range(len(p))))
    start = time.time()
    Fs = [item[2] for item in res if (item[0] == 2)]
    Fsj = [item[1][0] for item in res if (item[0] == 2)]
    Fsjw = [item[1] for item in res if (item[0] == 2)]
    Fsi = []
    tmp = -1
    for k in range(len(res)):
        if (res[k][0]):
            if (res[k][1][0] != tmp):
                tmp = res[k][1][0]
                Fsi.append(tmp)
    print ("analyze res list: ", time.time() - start)

    start = time.time()
    mins = findMins_2(p, q, r, s, Fs, Fsj, Fsi)
    print ("calc genl mins time: ", time.time() - start)

    _Aj = [Aj[j[0]] for j in Fsjw]
    _Bj = [Bj[j[0]] for j in Fsjw]
    _Aa = [Aa[j[0]] for j in Fsjw]
    _Ba = [Ba[j[0]] for j in Fsjw]
    __Aj = [Aj[j[1]] for j in Fsjw]
    __Bj = [Bj[j[1]] for j in Fsjw]
    __Aa = [Aa[j[1]] for j in Fsjw]
    __Ba = [Ba[j[1]] for j in Fsjw]
    if index is None:
        Fsjw = ['(' + str(j[0]) + ',' + str(j[1]) + ')' for j in Fsjw]
    else:
        Fsjw = ['(' + str(index[j[0]]) + ',' + str(index[j[1]]) + ')' for j in Fsjw]
    stratMinsData = pd.DataFrame({'Aj1':_Aj,'Bj1':_Bj,'Aa1':_Aa,'Ba1':_Ba,
                                    'Aj2':__Aj,'Bj2':__Bj,'Aa2':__Aa,'Ba2':__Ba,'min': mins}, index=Fsjw)
    idOptStrat = stratMinsData['min'].idxmax()
    return stratMinsData, idOptStrat

def fitMaxMin_2(stratData, pqrsData):
    if (stratData.index != pqrsData.index).all():
        print("WARNING: stratData.index != pqrsData.index")

    Aj = stratData['Aj'].tolist()
    Bj = stratData['Bj'].tolist()
    Aa = stratData['Aa'].tolist()
    Ba = stratData['Ba'].tolist()

    p = pqrsData['p'].values
    q = pqrsData['q'].values
    r = pqrsData['r'].values
    s = pqrsData['s'].values

    return genlFitMaxMin_2(Aj, Bj, Aa, Ba, p, q, r, s, index=pqrsData.index)

def checkRanking(stratPopFitData):
    tIdxs = stratPopFitData.sort_values(by=['t'], ascending=False).index.values
    fitIdxs = stratPopFitData.sort_values(by=['fit'], ascending=False).index.values

    count = 0
    _range = range(len(tIdxs))
    for i in _range:
        for j in _range:
            if (tIdxs[i] == fitIdxs[j]):
                count += np.abs(i-j)
                break

    return count

def checkMl(selData, coefData, idx=-1):
    y = selData['class']
    mpMatr = selData.loc[:, 'M1':'M8M8'].values
    lams = coefData.loc[idx].values

    fits = calcLinsum(mpMatr, lams)
    _y = []
    for fit in fits:
        if fit > 0:
            _y.append(1)
        elif fit < 0:
            _y.append(-1)
        else:
            print("WARNING: fits are equal?!")

    return (y == _y).mean()*100

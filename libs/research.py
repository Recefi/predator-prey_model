import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import tqdm

import libs.gen_selection as gs


def findFsols(p, q, r, s, left=-1000, right=1000, step=1, errEps = 1e-15, rndEps = 10, abs=True):
    Fsols = []
    for F0 in range(left, right, step):
        F, err = gs.calcFLim(p, q, r, s, F0, abs)
        if (np.abs(err) < errEps):
            F = np.round(F, rndEps)
            if not Fsols:
                Fsols.append(F)
            elif (np.abs(F - Fsols) > 10**(-rndEps)).all():
                Fsols.append(F)
    return Fsols

def chkFsols(p, q, r, s, Fsols):
    Flams = []
    lamsErrs = []
    for i in range(len(Fsols)):
        z1, z2 = gs.calcZLim(p, q, r, s, Fsols[i])
        roots, errs = gs.chkFLim(p, q, r, s, Fsols[i], z1, z2)
        if (roots.real > 0).any():
            Flams.append('+')
        else:
            Flams.append('-')
        lamsErrs.append(errs)
    return Flams, lamsErrs

def chkFsols_auto(p, q, r, s, Fsols):
    Flams = []
    for i in range(len(Fsols)):
        z1, z2 = gs.calcZLim(p, q, r, s, Fsols[i])
        roots = gs.chkFLim_auto(p, q, r, s, Fsols[i], z1, z2)
        if (roots.real > 0).any():
            Flams.append('+')
        else:
            Flams.append('-')
    return Flams

def chkFsolsOnSel(stratData, pqrsData, abs=True):
    p, q, r, s = (pqrsData[col] for col in pqrsData[['p','q','r','s']])

    res = []
    for i in tqdm.tqdm(pqrsData.index):
        resStr = []
        Fsols = findFsols(p[i], q[i], r[i], s[i],
                        left = -100, right=100,
                        step=1, abs=abs)
        Flams, lamsErrs = chkFsols(p[i], q[i], r[i], s[i], Fsols)
        lamsErrs = chkFsols_auto(p[i], q[i], r[i], s[i], Fsols)
        #resStr.append(Fsols)
        resStr.append(Flams)
        resStr.append(lamsErrs)
        res.append(resStr)
    FsolsData = pd.DataFrame(res, columns=['FlamsSigns', 'FlamsErrs'], index=pqrsData.index)
    FsolsData = pd.concat([stratData, FsolsData], axis=1)
    return FsolsData


def calcComplexFLim(p, q, r, s, reF0=0.1, imF0=0):
    def tmpFunc(F):
        return 2*r*p / (2*p*s + q*(-(p+q*F-s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r))) \
                                    - ((-(p+q*F+s*F) + np.emath.sqrt((p+q*F-s*F)**2 + 4*p*r)) / 2)**2 - F
    def complexFunc(F):
        re, im = F
        tmp = tmpFunc(complex(re, im))
        return [tmp.real, tmp.imag]
    
    root = fsolve(complexFunc, (reF0, imF0))
    F = complex(root[0], root[1])
    err = tmpFunc(F)
    # print(F)
    # print("err:", err)
    return F, err

def findComplexFsols(p, q, r, s, left=-1000, right=1000, step=1, errEps = 1e-15, reRndEps = 4, imRndEps = 4):
    Fsols = []
    for F0 in range(left, right, step):
        F, err = calcComplexFLim(p, q, r, s, F0, 1)
        if (np.abs(err.real) < errEps):
            F = np.round(F, reRndEps)
            if not Fsols:
                Fsols.append(F)
            elif((np.abs(F.real-np.real(Fsols)) > 10**(-reRndEps)).all()
            or (np.abs(F.imag-np.imag(Fsols)) > 10**(-imRndEps)).all()):
                Fsols.append(F)
    return Fsols

def chkComplexFsolsOnSel(stratData, pqrsData):
    p, q, r, s = (pqrsData[col] for col in pqrsData[['p','q','r','s']])

    res = []
    for i in tqdm.tqdm(pqrsData.index):
        resStr = []
        Fsols = findComplexFsols(p[i], q[i], r[i], s[i],
                                left = -100, right=100,
                                step=1)
        Flams, lamsErrs = chkFsols(p[i], q[i], r[i], s[i], Fsols)
        #resStr.append(Fsols)
        resStr.append(Flams)
        resStr.append(lamsErrs)
        res.append(resStr)
    complexFsolsData = pd.DataFrame(res, columns=['FlamsSigns', 'FlamsErrs'], index=pqrsData.index)
    complexFsolsData = pd.concat([stratData, complexFsolsData], axis=1)
    return complexFsolsData

def compareSearchFsols(stratData, pqrsData):
    p = pqrsData['p']
    q = pqrsData['q']
    r = pqrsData['r']
    s = pqrsData['s']
    _F01 = []
    _Fm100 = []
    _F100 = []
    _resF = []
    _integrF = []

    for j in tqdm.tqdm(pqrsData.index):
        F01, err = gs.calcFLim(p[j], q[j], r[j], s[j], F0=0.1)
        _F01.append(F01)
        Fm100, err = gs.calcFLim(p[j], q[j], r[j], s[j], F0=-100)
        _Fm100.append(Fm100)
        F100, err = gs.calcFLim(p[j], q[j], r[j], s[j], F0=100)
        _F100.append(F100)
        _resF.append(gs.findF(p, q, r, s, j))
        pqrsRow = pqrsData.loc[[j]]
        stratRow = stratData.loc[[j]]
        rawPopData = gs.calcPopDynamics(pqrsRow, tMax=500, tParts=1000, z0=0.001, F0=0.001, _method="Radau")
        stratPopData, integrF = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)
        _integrF.append(integrF)

    compareData = stratData.copy()
    compareData.loc[:, 'F_0.1'] = _F01
    compareData.loc[:, 'F_-100'] = _Fm100
    compareData.loc[:, 'F_100'] = _F100
    compareData.loc[:, 'F_res'] = _resF
    compareData.loc[:, 'F_integr'] = _integrF
    return compareData

def findFsols_2(p1,q1,r1,s1, p2,q2,r2,s2, left=-1000, right=1000, step=1, errEps = 1e-15, rndEps = 10, abs=True):
    Fsols = []
    for F0 in range(left, right, step):
        F, err = gs.calcFLim_2(p1,q1,r1,s1, p2,q2,r2,s2, F0, abs)
        if (np.abs(err) < errEps):
            F = np.round(F, rndEps)
            if not Fsols:
                Fsols.append(F)
            elif (np.abs(F - Fsols) > 10**(-rndEps)).all():
                Fsols.append(F)
    return Fsols

def chkFsols_2(p1,q1,r1,s1, p2,q2,r2,s2, Fsols):
    FlamsSign = []
    Flams = []
    for i in range(len(Fsols)):
        z1, z2, z3, z4 = gs.calcZLim_2(p1,q1,r1,s1, p2,q2,r2,s2, Fsols[i])
        roots = gs.chkFLim_2(p1,q1,r1,s1, p2,q2,r2,s2, Fsols[i], z1, z2, z3, z4)
        if (roots.real > 0).any():
            FlamsSign.append('+')
        else:
            FlamsSign.append('-')
        Flams.append(roots.tolist())
    return FlamsSign, Flams

def chkFsolsOnSel_2(stratData, pqrsData, abs=True):
    p, q, r, s = (pqrsData[col] for col in pqrsData[['p','q','r','s']])

    res = []
    idxs = []
    for i in tqdm.tqdm(pqrsData.index):
        for j in pqrsData.index:
            if i == j:
                continue
            resStr = []
            Fsols = findFsols_2(p[i], q[i], r[i], s[i], p[j], q[j], r[j], s[j],
                                                        left = -100, right=100,
                                                        step=1, abs=abs)
            FlamsSign, Flams = chkFsols_2(p[i], q[i], r[i], s[i], p[j], q[j], r[j], s[j], Fsols)
            resStr.append(Fsols)
            resStr.append(FlamsSign)
            resStr.append(Flams)
            idxs.append('('+str(i)+','+str(j)+')')
            res.append(resStr)
    FsolsData = pd.DataFrame(res, columns=['Fsols', 'FlamsSign', 'Flams'], index=idxs)
    return FsolsData

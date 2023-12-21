import pandas as pd
import numpy as np
from numpy import sqrt
from sklearn.metrics.pairwise import cosine_similarity

import libs.param as param


a_j = param.alpha_j
g_j = param.gamma_j
b_j = param.beta_j
d_j = param.delta_j
a_a = param.alpha_a
g_a = param.gamma_a
b_a = param.beta_a
d_a = param.delta_a

def getDerivatives(p, q, r, s):
    """Считаем частные производные в конкретной точке (p,q,r,s)"""
    hp = -1 + (4*r + 2*(p + q - s))/(2*sqrt(4*p*r + (p + q - s)**2))
    hq = -1 + (p + q - s)/sqrt(4*p*r + (p + q - s)**2)
    hr = (2*p)/sqrt(4*p*r + (p + q - s)**2)
    hs = -1 - (p + q - s)/sqrt(4*p*r + (p + q - s)**2)
    hpp = -(4*r*(q + r - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hpq = (2*r*(p - q + s))/(4*p*r + (p + q - s)**2)**(3/2)
    hpr = (2*((q - s)**2 + p*(q + 2*r - s)))/(4*p*r + (p + q - s)**2)**(3/2)
    hps = -(2*r*(p - q + s))/(4*p*r + (p + q - s)**2)**(3/2)
    hqq = (4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)
    hqr = -(2*p*(p + q - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hqs = -(4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)
    hrr = -(4*p**2)/(4*p*r + (p + q - s)**2)**(3/2)
    hrs = (2*p*(p + q - s))/(4*p*r + (p + q - s)**2)**(3/2)
    hss = (4*p*r)/(4*p*r + (p + q - s)**2)**(3/2)

    return hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss

def denormMlLams(norm_mlLams, selData):
    """
    lam0 у "разнормированных" лямбд не имеет физичемкого смысла, но тем не менее здесь она сохраняется,
        т.к. гипотетически может быть использована для построения графиков проекций по исходным разностям и оценки погрешности, которая из-за lam0 возникает...
    """
    maxs = selData.loc[-1,'M1':'M8M8'].to_list()
    mlLams = [norm_mlLams[0]]
    for i in range(44):
        mlLams.append(norm_mlLams[i+1] / maxs[i])
    return mlLams

def normCalcLams(calcLams, selData):
    maxs = selData.loc[-1,'M1':'M8M8'].to_list()
    norm_calcLams = []
    for i in range(44):
        mlLams.append(norm_mlLams[i] * maxs[i])
    return norm_calcLams

def getCoefData(pqrsData, norm_mlLams, mlLams):
    """Считаем коэф-ты для всех точек (p,q,r,s) по сокращенной формуле"""
    lamCol = []
    for i in range(1,9):
        lamCol.append('lam'+str(i))
    for i in range(1,9):
        for j in range(i,9):
            lamCol.append('lam'+str(i)+str(j))
    
    coefTable = []
    coefTable.append(norm_mlLams)
    coefTable.append(mlLams)
    for i in pqrsData.index:
        p, q, r, s = pqrsData.loc[i]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p, q, r, s)

        # Считаем коэффициенты разложения в данной точке (по строкам: при M1-M8, M11-M18, M22-M28, M33-M38, ..., M88)
        calcCoefs = [hp*a_j, hq*(-g_j), hp*b_j, hp*d_j, hr*a_a, hs*(-g_a), hr*b_a, hr*d_a,
                    1/2*hpp*a_j**2, 1/2*hpq*a_j*(-g_j), 1/2*hpp*2*a_j*b_j, 1/2*hpp*2*a_j*d_j, 1/2*hpr*a_j*a_a, 1/2*hps*a_j*(-g_a), 1/2*hpr*a_j*b_a, 1/2*hpr*a_j*d_a,
                    1/2*hqq*(-g_j)**2, 1/2*hpq*b_j*(-g_j), 1/2*hpq*d_j*(-g_j), 1/2*hqr*(-g_j)*a_a, 1/2*hqs*(-g_j)*(-g_a), 1/2*hqr*(-g_j)*b_a, 1/2*hqr*(-g_j)*d_a,
                    1/2*hpp*b_j**2, 1/2*hpp*2*b_j*d_j, 1/2*hpr*b_j*a_a, 1/2*hps*b_j*(-g_a), 1/2*hpr*b_j*b_a, 1/2*hpr*b_j*d_a,
                    1/2*hpp*d_j**2, 1/2*hpr*d_j*a_a, 1/2*hps*d_j*(-g_a), 1/2*hpr*d_j*b_a, 1/2*hpr*d_j*d_a,
                    1/2*hrr*a_a**2, 1/2*hrs*a_a*(-g_a), 1/2*hrr*2*a_a*b_a, 1/2*hrr*2*a_a*d_a,
                    1/2*hss*(-g_a)**2, 1/2*hrs*b_a*(-g_a), 1/2*hrs*d_a*(-g_a),
                    1/2*hrr*b_a**2, 1/2*hrr*2*b_a*d_a,
                    1/2*hrr*d_a**2]

        coefTable.append(calcCoefs)
    
    indexes = [-2, -1]
    indexes.extend(pqrsData.index)
    coefData = pd.DataFrame(coefTable, columns=lamCol, index=indexes)
    return coefData

def getCosinesCoef(coefData):
    "Косинусы между векторами вычисленных и вектором машинных коэффициентов"
    cosines = cosine_similarity(coefData)[1]
    print("\nвсе косинусы:\n", cosines, "\n")
    print("\nнеобходимые косинусы:\n\n", cosines[2:], "\n")

    cosines = pd.Series(cosines[2:])
    cosines.index = coefData.index[2:]
    return cosines

def checkCoef(coefData, fitData, mpData, pqrsData, maxFitPntId, nearPntId):
    """Сравнение способов восстановления функции фитнеса"""
    def taylor(pntId):
        p0, q0, r0, s0 = pqrsData.loc[pntId]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0)
        fit0 = fitData.loc[pntId, 'fit']

        taylorFit = []
        for i in pqrsData.index:
            p, q, r, s = pqrsData.loc[i]
            taylorFit.append(fit0 + hp*(p-p0) + hq*(q-q0) + hr*(r-r0) + hs*(s-s0) + 1/2*(hpp*(p-p0)**2 + hqq*(q-q0)**2 + hrr*(r-r0)**2 + hss*(s-s0)**2 + hpq*(p-p0)*(q-q0) + hpr*(p-p0)*(r-r0) + hps*(p-p0)*(s-s0) + hqr*(q-q0)*(r-r0) + hqs*(q-q0)*(s-s0) + hrs*(r-r0)*(s-s0)))
        return taylorFit

    def restoreFits(pntId):
        fits = []
        lams = coefData.loc[pntId].to_list()
        for i in stratFitData.index:
            Mp = mpData.loc[i, 'M1':'M8M8'].to_list()
            fit = 0
            for j in range(44):
                fit += lams[j]*Mp[j]
            fits.append(fit)
        return fits
    
    trueFits = fitData['fit']
    T_maxFitPnt = taylor(maxFitPntId)
    # rF_maxFitPnt = restoreFits(maxFitPntId)
    T_nearPnt = taylor(nearPntId)
    # rF_nearPnt = restoreFits(nearPntId)
    restoredFits_norm = restoreFits(-2)
    restoredFits = restoreFits(-1)

    # checkCoefData = pd.DataFrame({'trueFit': trueFits, 'T_maxFitPnt': T_maxFitPnt, 'rF_maxFitPnt': rF_maxFitPnt,
    #                               'T_nearPnt': T_nearPnt, 'rF_nearPnt': rF_nearPnt, "restoredFit_norm": restoredFits_norm, "restoredFit": restoredFits}, index=trueFits.index)
    checkCoefData = pd.DataFrame({'trueFit': trueFits, 'T_maxFitPnt': T_maxFitPnt, 'T_nearPnt': T_nearPnt, "lam_norm": restoredFits_norm, "lam": restoredFits}, index=trueFits.index)    
    return checkCoefData

def getFitCosines(checkCoefData):
    cosines = cosine_similarity(checkCoefData.transpose())[0]
    return cosines

def compareCoefs(coefData, nearPntId, maxFitId):
    """Сравниваем коэффициенты, нормируя по первому столбцу"""
    mlLams = coefData.loc[-1].copy()
    calcCoefs_nearPnt = coefData.loc[nearPntId].copy()
    calcCoefs_maxFitPnt = coefData.loc[maxFitId].copy()
    
    # mlLams/=(np.max(np.abs(mlLams)))
    # calcCoefs_nearPnt/=(np.max(np.abs(calcCoefs_nearPnt)))
    # calcCoefs_maxFitPnt/=(np.max(np.abs(calcCoefs_maxFitPnt)))

    mlLams/=(np.abs(mlLams[0]))
    calcCoefs_nearPnt/=(np.abs(calcCoefs_nearPnt[0]))
    calcCoefs_maxFitPnt/=(np.abs(calcCoefs_maxFitPnt[0]))   

    compareCoefData = pd.DataFrame({'machine': mlLams, 'nearPnt': calcCoefs_nearPnt, 'maxFitPnt': calcCoefs_maxFitPnt}, index=coefData.columns)
    return compareCoefData


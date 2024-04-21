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

def denormMlLams(norm_mlLams, maxs):
    """
    lam0 у "разнормированных" лямбд не имеет физического смысла, но тем не менее здесь она сохраняется,
        т.к. гипотетически может быть использована для построения графиков проекций по исходным разностям и оценки погрешности, которая из-за lam0 возникает...
    """
    mlLams = [norm_mlLams[0]]
    for i in range(44):
        mlLams.append(norm_mlLams[i+1] / maxs[i])
    return mlLams

def normCalcLams(calcLams, maxs):
    norm_calcLams = []
    for i in range(44):
        norm_calcLams.append(calcLams[i] * maxs[i])
    return norm_calcLams

def getDerivatives(p, q, r, s, F=1):
    """Считаем частные производные в конкретной точке (p,q,r,s)"""
    hp = -1 + (4*r + 2*(p + q*F - s*F))/(2*sqrt(4*p*r + (p + q*F - s*F)**2))
    hq = -F + F*(p + q*F - s*F)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hr = (2*p)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hs = -F - F*(p + q*F - s*F)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hpp = -(4*r*(q*F + r - s*F))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hpq = (2*F*r*(p - q*F + s*F))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hpr = (2*(F**2 * (q - s)**2 + p*(q*F + 2*r - s*F)))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hps = -(2*F*r*(p - q*F + s*F))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hqq = (4*F*F*p*r)/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hqr = -(2*F*p*(p + q*F - s*F))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hqs = -(4*F*F*p*r)/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hrr = -(4*p**2)/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hrs = (2*F*p*(p + q*F - s*F))/(4*p*r + (p + q*F - s*F)**2)**(3/2)
    hss = (4*F*F*p*r)/(4*p*r + (p + q*F - s*F)**2)**(3/2)

    return hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss

def getCoefData(pqrsData, norm_mlLams, mlLams, F=1):
    """Считаем коэф-ты для всех точек (p,q,r,s)"""
    lamCol = []
    for i in range(1,9):
        lamCol.append('lam'+str(i))
    for i in range(1,9):
        for j in range(i,9):
            lamCol.append('lam'+str(i)+str(j))
    
    indexes = [-2, -1]
    coefTable = []
    coefTable.append(norm_mlLams)
    coefTable.append(mlLams)
    for i in pqrsData.index:
        p, q, r, s = pqrsData.loc[i]
        if (4*p*r + (p + q*F - s*F)**2 < 0):
            continue
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p, q, r, s, F)

        # Считаем коэффициенты разложения в данной точке (по строкам: при M1-M8, M11-M18, M22-M28, M33-M38, ..., M88)
        calcCoefs = [
        hp*a_j, hq*(-g_j), hp*b_j, hp*d_j, hr*a_a, hs*(-g_a), hr*b_a, hr*d_a,
        1/2*hpp*a_j**2, 1/2*hpq*a_j*(-g_j), 1/2*hpp*2*a_j*b_j, 1/2*hpp*2*a_j*d_j, 1/2*hpr*a_j*a_a, 1/2*hps*a_j*(-g_a), 1/2*hpr*a_j*b_a, 1/2*hpr*a_j*d_a,
        1/2*hqq*(-g_j)**2, 1/2*hpq*b_j*(-g_j), 1/2*hpq*d_j*(-g_j), 1/2*hqr*(-g_j)*a_a, 1/2*hqs*(-g_j)*(-g_a), 1/2*hqr*(-g_j)*b_a, 1/2*hqr*(-g_j)*d_a,
        1/2*hpp*b_j**2, 1/2*hpp*2*b_j*d_j, 1/2*hpr*b_j*a_a, 1/2*hps*b_j*(-g_a), 1/2*hpr*b_j*b_a, 1/2*hpr*b_j*d_a,
        1/2*hpp*d_j**2, 1/2*hpr*d_j*a_a, 1/2*hps*d_j*(-g_a), 1/2*hpr*d_j*b_a, 1/2*hpr*d_j*d_a,
        1/2*hrr*a_a**2, 1/2*hrs*a_a*(-g_a), 1/2*hrr*2*a_a*b_a, 1/2*hrr*2*a_a*d_a,
        1/2*hss*(-g_a)**2, 1/2*hrs*b_a*(-g_a), 1/2*hrs*d_a*(-g_a),
        1/2*hrr*b_a**2, 1/2*hrr*2*b_a*d_a,
        1/2*hrr*d_a**2
        ]

        indexes.append(i)
        coefTable.append(calcCoefs)
    
    coefData = pd.DataFrame(coefTable, columns=lamCol, index=indexes)
    return coefData

def getCosinesCoef(coefData):
    "Косинусы между векторами вычисленных и вектором машинных коэффициентов"
    cosines = cosine_similarity(coefData)[1]
    #print("\nвсе косинусы:\n\n", cosines, "\n")
    #print("\nкосинусы между машинными и вычисленными:\n\n", cosines[2:], "\n")

    cosines = pd.Series(cosines[2:])
    cosines.index = coefData.index[2:]
    return cosines


def compareCoefs_static(coefData, nearPntId, maxFitId):
    """Сравниваем коэффициенты, нормируя по lam1"""
    norm_mlLams = coefData.loc[-2].copy()
    mlLams = coefData.loc[-1].copy()
    calcLams_nearPnt = coefData.loc[nearPntId].copy()
    calcLams_maxFitPnt = coefData.loc[maxFitId].copy()

    norm_mlLams/=(np.abs(norm_mlLams.loc['lam1']))
    mlLams/=(np.abs(mlLams.loc['lam1']))
    calcLams_nearPnt/=(np.abs(calcLams_nearPnt.loc['lam1']))
    calcLams_maxFitPnt/=(np.abs(calcLams_maxFitPnt.loc['lam1']))   

    compareCoefData = pd.DataFrame({'norm_machine': norm_mlLams, 'machine': mlLams,
                                'nearPnt': calcLams_nearPnt, 'maxFitPnt': calcLams_maxFitPnt}, index=coefData.columns)
    compareCoefData.loc['cosines'] = cosine_similarity(compareCoefData.transpose())[1]
    return compareCoefData

def compareCoefs_dynamic(coefData, nearPntId):
    """Сравниваем коэффициенты, нормируя по lam1"""
    norm_mlLams = coefData.loc[-2].copy()
    mlLams = coefData.loc[-1].copy()
    calcLams_nearPnt = coefData.loc[nearPntId].copy()

    norm_mlLams/=(np.abs(norm_mlLams.loc['lam1']))
    mlLams/=(np.abs(mlLams.loc['lam1']))
    calcLams_nearPnt/=(np.abs(calcLams_nearPnt.loc['lam1']))

    compareCoefData = pd.DataFrame({'norm_machine': norm_mlLams, 'machine': mlLams,
                                'nearPnt': calcLams_nearPnt}, index=coefData.columns)
    compareCoefData.loc['cosines'] = cosine_similarity(compareCoefData.transpose())[1]
    return compareCoefData


def compareFits_static(coefData, fitData, mpData, pqrsData, maxFitPntId, nearPntId):
    """Сравнение способов восстановления функции фитнеса"""
    F=1
    def taylor(pntId):
        p0, q0, r0, s0 = pqrsData.loc[pntId]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0, F)
        fit0 = fitData.loc[pntId, 'fit']

        taylorFit = []
        for i in coefData.loc[0:].index:
            p, q, r, s = pqrsData.loc[i]
            taylorFit.append(fit0 + hp*(p-p0) + F*hq*(q-q0) + hr*(r-r0) + F*hs*(s-s0)
                                + 1/2*(hpp*(p-p0)**2 + F*F*hqq*(q-q0)**2 + hrr*(r-r0)**2 + F*F*hss*(s-s0)**2
                                            + F*hpq*(p-p0)*(q-q0) + hpr*(p-p0)*(r-r0) + F*hps*(p-p0)*(s-s0)
                                                + F*hqr*(q-q0)*(r-r0) + F*F*hqs*(q-q0)*(s-s0) + F*hrs*(r-r0)*(s-s0)))
        return taylorFit

    def restoreFits(pntId):
        fits = []
        lams = coefData.loc[pntId, 'lam1':'lam88'].to_list()
        for i in fitData.index:
            Mp = mpData.loc[i, 'M1':'M8M8'].to_list()
            fit = 0
            for j in range(44):
                fit += lams[j]*Mp[j]
            fits.append(fit)
        return fits
    
    trueFits = fitData['fit']
    maxFitPnt_T = taylor(maxFitPntId)
    nearPnt_T = taylor(nearPntId)
    maxFitPnt_lam = restoreFits(maxFitPntId)
    nearPnt_lam = restoreFits(nearPntId)
    ml_lam_norm = restoreFits(-2)
    ml_lam = restoreFits(-1)

    # compareFitData = pd.DataFrame({'trueFit': trueFits, 'maxFitPnt_T': maxFitPnt_T, 'nearPnt_T': nearPnt_T,
    #                                                   "lam_norm": ml_lam_norm, "lam": ml_lam}, index=trueFits.index)
    compareFitData = pd.DataFrame({'trueFit': trueFits, 'maxFitPnt_T': maxFitPnt_T, 'nearPnt_T': nearPnt_T,
                                                'maxFitPnt_lam': maxFitPnt_lam, 'nearPnt_lam': nearPnt_lam,
                                                    "ml_lam_norm": ml_lam_norm, "ml_lam": ml_lam}, index=trueFits.index)    
    cosines = cosine_similarity(compareFitData.transpose())[0]
    return compareFitData, cosines

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
    """lam0 у "разнормированных" лямбд не имеет физического смысла, но тем не менее здесь она сохраняется"""
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
    """legacy"""
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

def calcDers_qFsF(p, q, r, s, F=1):
    """calc h by qF,sF"""
    hp = -1 + (4*r + 2*(p + q*F - s*F))/(2*sqrt(4*p*r + (p + q*F - s*F)**2))
    hq = -1 + (p + q*F - s*F)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hr = (2*p)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hs = -1 - (p + q*F - s*F)/sqrt(4*p*r + (p + q*F - s*F)**2)
    hpp = -(4*r*(q*F + r - s*F))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hpq = (2*r*(p - q*F + s*F))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hpr = (2*(F**2 * (q - s)**2 + p*(q*F + 2*r - s*F)))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hps = -(2*r*(p - q*F + s*F))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hqq = (4*p*r)/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hqr = -(2*p*(p + q*F - s*F))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hqs = -(4*p*r)/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hrr = -(4 * p**2)/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hrs = (2*p*(p + q*F - s*F))/((4*p*r + (p + q*F - s*F)**2)**(3/2))
    hss = (4*p*r)/((4*p*r + (p + q*F - s*F)**2)**(3/2))

    return hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss

def imputeDers_qFsF(h, F):
    """impute ders of qF,sF to h"""
    hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = h
    hq *= F
    hs *= F
    hpq *= F
    hps *= F
    hqq *= F*F
    hqr *= F
    hqs *= F*F
    hrs *= F
    hss *= F*F
    return hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss

def calcDers_qs(p, q, r, s, F=1):
    """calc h by q,s"""
    return imputeDers_qFsF(calcDers_qFsF(p, q, r, s, F), F)

def calcCoefs(h, params=(param.alpha_j, param.beta_j, param.gamma_j, param.delta_j,
                        param.alpha_a, param.beta_a, param.gamma_a, param.delta_a)):
    hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = h
    a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = params

    _p = hp
    _q = hq
    _r = hr
    _s = hs
    _pp, _qq, _rr, _ss = 1/2*hpp, 1/2*hqq, 1/2*hrr, 1/2*hss
    _pq, _pr, _ps, _qr, _qs, _rs = hpq, hpr, hps, hqr, hqs, hrs

    # Считаем коэффициенты разложения в данной точке (по строкам: при M1-M8, M11-M18, M22-M28, M33-M38, ..., M88)
    return (
    _p*a_j, _q*(-g_j), _p*b_j, _p*d_j, _r*a_a, _s*(-g_a), _r*b_a, _r*d_a,
    _pp*a_j**2, _pq*a_j*(-g_j), _pp*2*a_j*b_j, _pp*2*a_j*d_j, _pr*a_j*a_a, _ps*a_j*(-g_a), _pr*a_j*b_a, _pr*a_j*d_a,
    _qq*(-g_j)**2, _pq*b_j*(-g_j), _pq*d_j*(-g_j), _qr*(-g_j)*a_a, _qs*(-g_j)*(-g_a), _qr*(-g_j)*b_a,_qr*(-g_j)*d_a,
    _pp*b_j**2, _pp*2*b_j*d_j, _pr*b_j*a_a, _ps*b_j*(-g_a), _pr*b_j*b_a, _pr*b_j*d_a,
    _pp*d_j**2, _pr*d_j*a_a, _ps*d_j*(-g_a), _pr*d_j*b_a, _pr*d_j*d_a,
    _rr*a_a**2, _rs*a_a*(-g_a), _rr*2*a_a*b_a, _rr*2*a_a*d_a,
    _ss*(-g_a)**2, _rs*b_a*(-g_a), _rs*d_a*(-g_a),
    _rr*b_a**2, _rr*2*b_a*d_a,
    _rr*d_a**2
    )

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

        h = calcDers_qs(p, q, r, s, F)  # TODO: проверить на 3 выборке на == getDerivatives + на статическом проверить еще == calcDers_qFsF
        coefTable.append(calcCoefs(h))
        indexes.append(i)

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

def compareCoefs(coefData, nearPntId, optPntId):
    """Сравниваем коэффициенты, нормируя по lam1"""
    norm_mlLams = coefData.loc[-2].copy()
    mlLams = coefData.loc[-1].copy()
    calcLams_nearPnt = coefData.loc[nearPntId].copy()
    calcLams_optPnt = coefData.loc[optPntId].copy()

    norm_mlLams/=(np.abs(norm_mlLams.loc['lam1']))
    mlLams/=(np.abs(mlLams.loc['lam1']))
    calcLams_nearPnt/=(np.abs(calcLams_nearPnt.loc['lam1']))
    calcLams_optPnt/=(np.abs(calcLams_optPnt.loc['lam1']))

    compareCoefData = pd.DataFrame({'norm_machine': norm_mlLams, 'machine': mlLams,
                                'nearPnt': calcLams_nearPnt, 'optPnt': calcLams_optPnt}, index=coefData.columns)
    compareCoefData.loc['cosines'] = cosine_similarity(compareCoefData.transpose())[1]
    return compareCoefData

def compareFits(coefData, fitData, mpData, pqrsData, nearPntId, optPntId, F=1):
    """Оценка восстановления функции фитнеса по самим значениям функции"""
    def taylor(pntId):
        p0, q0, r0, s0 = pqrsData.loc[pntId]
        hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = getDerivatives(p0, q0, r0, s0, F)
        fit0 = fitData.loc[pntId, 'fit']

        taylorFit = []
        for i in coefData.loc[0:].index:
            p, q, r, s = pqrsData.loc[i]
            taylorFit.append(fit0 + hp*(p-p0) + hq*(q-q0) + hr*(r-r0) + hs*(s-s0)
                                + 1/2*(hpp*(p-p0)**2 + hqq*(q-q0)**2 + hrr*(r-r0)**2 + hss*(s-s0)**2
                                            + hpq*(p-p0)*(q-q0) + hpr*(p-p0)*(r-r0) + hps*(p-p0)*(s-s0)
                                                + hqr*(q-q0)*(r-r0) + hqs*(q-q0)*(s-s0) + hrs*(r-r0)*(s-s0)))
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
    optPnt_T = taylor(optPntId)
    nearPnt_T = taylor(nearPntId)
    optPnt_lam = restoreFits(optPntId)
    nearPnt_lam = restoreFits(nearPntId)
    ml_lam_norm = restoreFits(-2)
    ml_lam = restoreFits(-1)

    # compareFitData = pd.DataFrame({'trueFit': trueFits, 'optPnt_lam': optPnt_lam,'nearPnt_lam': nearPnt_lam,
    #                                                   "lam_norm": ml_lam_norm, "lam": ml_lam}, index=trueFits.index)
    compareFitData = pd.DataFrame({'trueFit': trueFits, 'optPnt_taylor': optPnt_T, 'optPnt_lam': optPnt_lam,
                                                'nearPnt_taylor': nearPnt_T, 'nearPnt_lam': nearPnt_lam,
                                                    "ml_lam_norm": ml_lam_norm, "ml_lam": ml_lam}, index=trueFits.index)    
    cosines = cosine_similarity(compareFitData.transpose())[0]
    return compareFitData, cosines

import numpy as np
import pandas as pd
import copy
import tqdm
import sys
import signal

import libs.param as param
import libs.gen_selection as gs
import libs.taylor as tr


def compareRestoredPQRS(p, q, r, s, pqrsData, optPntId):
    comparePqrsData = pd.DataFrame(columns=['p','q','r','s'])
    comparePqrsData.loc['true'] = pqrsData.loc[optPntId, ['p','q','r','s']]
    comparePqrsData.loc['restored'] = [p, q, r, s]
    return comparePqrsData

def compareRestoredParam(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    compareParamData = pd.DataFrame(columns=['a_j','b_j','g_j','d_j','a_a','b_a','g_a','d_a'])
    compareParamData.loc['true'] = [param.alpha_j, param.beta_j, param.gamma_j, param.delta_j,
                                        param.alpha_a, param.beta_a, param.gamma_a, param.delta_a]
    compareParamData.loc['restored'] = [a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a]
    return compareParamData

def restorePQRS_1(FLim, stratPopData, coefData, mpData, optPntId, lamsKey=-1):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    M2 = mpData.loc[optPntId, 'M2']
    M6 = mpData.loc[optPntId, 'M6']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam66 = coefData.loc[lamsKey, 'lam66']

    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim
    q = lam26*M2/(lam26*M2*z1Lim - 2*lam66*M6*z2Lim)  # (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z1Lim-(2*lam66*M6*z2Lim)/(lam26*M2)))
    s = -(2*lam66*M6*q)/(lam26*M2)
    p = (-q*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim

    return p, q, r, s

def restorePQRS_2(FLim, stratPopData, coefData, mpData, optPntId, lamsKey=-1):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    M2 = mpData.loc[optPntId, 'M2']
    M6 = mpData.loc[optPntId, 'M6']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam22 = coefData.loc[lamsKey, 'lam22']

    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim
    s = lam26*M6/(lam26*M6*z2Lim - 2*lam22*M2*z1Lim)  # (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z2Lim-(2*lam22*M2*z1Lim)/(lam26*M6)))
    q = -(2*lam22*M2*s)/(lam26*M6)
    p = (-q*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim

    return p, q, r, s

def restorePQRS_3(FLim, stratPopData, coefData, mpData, optPntId, lamsKey=-1):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    M2 = mpData.loc[optPntId, 'M2']
    M6 = mpData.loc[optPntId, 'M6']
    lam22 = coefData.loc[lamsKey, 'lam22']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam66 = coefData.loc[lamsKey, 'lam66']

    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim

    q1 = (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z1Lim-(2*lam66*M6*z2Lim)/(lam26*M2)))
    s1 = -(2*lam66*M6*q1)/(lam26*M2)
    p1 = (-q1*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim
    #p1_2 = (s1*FLim*z2Lim + z2Lim*(z1Lim + z2Lim))/z1Lim

    s2 = (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z2Lim-(2*lam22*M2*z1Lim)/(lam26*M6)))
    q2 = -(2*lam22*M2*s2)/(lam26*M6)
    p2 = (-q2*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim
    #p2_2 = (s2*FLim*z2Lim + z2Lim*(z1Lim + z2Lim))/z1Lim

    p = (p1 + p2)/2
    #print(p1, p2, p1_2, p2_2)
    q = (q1 + q2)/2
    s = (s1 + s2)/2

    return p, q, r, s

def checkPqrs(mpData, optPntId, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    p = a_j*M1 + b_j*M3 + d_j*M4
    r = a_a*M5 + b_a*M7 + d_a*M8
    q = -g_j*M2
    s = -g_a*M6
    return p,q,r,s

# TODO: check it out
def qzsz_1(q, s, stratPopData, optPntId, z1Lim=None, z2Lim=None):
    if(z1Lim is None):
        z1Lim = stratPopData.loc[optPntId, 'z1']
    if(z2Lim is None):
        z2Lim = stratPopData.loc[optPntId, 'z2']
    return q*z1Lim+s*z2Lim

# TODO: check it out
def qzsz_2(r, FLim, stratPopData, optPntId, z1Lim=None, z2Lim=None):
    if(z1Lim is None):
        z1Lim = stratPopData.loc[optPntId, 'z1']
    if(z2Lim is None):
        z2Lim = stratPopData.loc[optPntId, 'z2']
    return (r*z2Lim - (z1Lim+z2Lim)**2)/FLim

def restoreParam(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[lamsKey, 'lam1':'lam8']

    g_j = -q/M2
    g_a = -s/M6

    a_j = lam1*p/(lam1*M1+lam3*M3+lam4*M4)
    b_j = lam3*p/(lam1*M1+lam3*M3+lam4*M4)
    d_j = lam4*p/(lam1*M1+lam3*M3+lam4*M4)

    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)
    b_a = lam7*r/(lam5*M5+lam7*M7+lam8*M8)
    d_a = lam8*r/(lam5*M5+lam7*M7+lam8*M8)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_2(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1,lam3,lam4,lam11,lam12,lam13,lam14,lam15,lam16,lam17,lam18,lam23,lam24,lam33,lam34,lam35,lam36,lam37,lam38,
    lam44,lam45,lam46,lam47,lam48) = coefData.loc[lamsKey, ['lam1','lam3','lam4','lam11','lam12','lam13','lam14','lam15',
    'lam16','lam17','lam18','lam23','lam24','lam33','lam34','lam35','lam36','lam37','lam38',
    'lam44','lam45','lam46','lam47','lam48']]

    a_j = lam1*p/(lam1*M1+lam3*M3+lam4*M4)

    b_j = lam13*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    d_j = lam14*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))

    a_a = lam15*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_a = lam17*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    d_a = lam18*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))

    g_j = -lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    g_a = -lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_4(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8) = coefData.loc[lamsKey,
    ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8']]
    (lam11, lam12, lam13, lam14, lam15, lam16, lam17, lam18) = coefData.loc[lamsKey,
    ['lam11','lam12','lam13','lam14','lam15','lam16','lam17','lam18']]
    (lam22, lam23, lam24, lam25, lam26, lam27, lam28) = coefData.loc[lamsKey,
    ['lam22','lam23','lam24','lam25','lam26','lam27','lam28']]
    (lam33, lam34, lam35, lam36, lam37, lam38) = coefData.loc[lamsKey,
    ['lam33','lam34','lam35','lam36','lam37','lam38']]
    (lam44, lam45, lam46, lam47, lam48) = coefData.loc[lamsKey,
    ['lam44','lam45','lam46','lam47','lam48']]
    (lam55, lam56, lam57, lam58) = coefData.loc[lamsKey,
    ['lam55','lam56','lam57','lam58']]
    (lam66, lam67, lam68) = coefData.loc[lamsKey,
    ['lam66','lam67','lam68']]
    (lam77, lam78, lam88) = coefData.loc[lamsKey,
    ['lam77','lam78','lam88']]

    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)
    print(a_a)

    _a_j = []
    _a_j.append(lam1*p/(lam1*M1+lam3*M3+lam4*M4))
    _a_j.append(lam15*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    a_j = np.mean(_a_j)
    print(_a_j)

    _d_a = []
    _d_a.append(lam8*r/(lam5*M5+lam7*M7+lam8*M8))
    _d_a.append(lam58*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _d_a.append(lam18*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    d_a = np.mean(_d_a)
    print(_d_a)

    _d_j = []
    _d_j.append(lam4*p/(lam1*M1+lam3*M3+lam4*M4))
    _d_j.append(lam45*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _d_j.append(lam14*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _d_j.append(lam48*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    d_j = np.mean(_d_j)
    print(_d_j)

    _b_a = []
    _b_a.append(lam7*r/(lam5*M5+lam7*M7+lam8*M8))
    _b_a.append(lam57*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _b_a.append(lam17*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_a.append(lam78*r*r/(2*d_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _b_a.append(lam47*p*r/(d_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    b_a = np.mean(_b_a)
    print(_b_a)

    _b_j = []
    _b_j.append(lam3*p/(lam1*M1+lam3*M3+lam4*M4))
    _b_j.append(lam35*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_j.append(lam13*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _b_j.append(lam38*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_j.append(lam34*p*p/(2*d_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _b_j.append(lam37*p*r/(b_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    b_j = np.mean(_b_j)
    print(_b_j)

    _g_a = []
    _g_a.append(-s/M6) #
    _g_a.append(-lam56*r*s/(a_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)))
    _g_a.append(-lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    _g_a.append(-lam68*r*s/(d_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))) #
    _g_a.append(-lam46*p*s/(d_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    _g_a.append(-lam67*r*s/(b_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)))
    _g_a.append(-lam36*p*s/(b_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    g_a = np.mean(_g_a)
    print(_g_a)

    _g_j = []
    _g_j.append(-q/M2) #
    _g_j.append(-lam25*q*r/(a_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)))
    _g_j.append(-lam28*q*r/(d_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam24*p*q/(d_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))) #
    _g_j.append(-lam27*q*r/(b_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam23*p*q/(b_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)))
    _g_j.append(q*s/(g_a*M2*M6))
    g_j = np.mean(_g_j)
    print(_g_j)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_5(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8) = coefData.loc[lamsKey,
    ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8']]
    (lam11, lam12, lam13, lam14, lam15, lam16, lam17, lam18) = coefData.loc[lamsKey,
    ['lam11','lam12','lam13','lam14','lam15','lam16','lam17','lam18']]
    (lam22, lam23, lam24, lam25, lam26, lam27, lam28) = coefData.loc[lamsKey,
    ['lam22','lam23','lam24','lam25','lam26','lam27','lam28']]
    (lam33, lam34, lam35, lam36, lam37, lam38) = coefData.loc[lamsKey,
    ['lam33','lam34','lam35','lam36','lam37','lam38']]
    (lam44, lam45, lam46, lam47, lam48) = coefData.loc[lamsKey,
    ['lam44','lam45','lam46','lam47','lam48']]
    (lam55, lam56, lam57, lam58) = coefData.loc[lamsKey,
    ['lam55','lam56','lam57','lam58']]
    (lam66, lam67, lam68) = coefData.loc[lamsKey,
    ['lam66','lam67','lam68']]
    (lam77, lam78, lam88) = coefData.loc[lamsKey,
    ['lam77','lam78','lam88']]

    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)
    print(a_a)

    a_j1 = lam1*p/(lam1*M1+lam3*M3+lam4*M4)
    a_j2 = lam15*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    a_j = (a_j1 + a_j2)/2
    print(a_j1, a_j2)

    d_a1 = lam8*r/(lam5*M5+lam7*M7+lam8*M8)
    d_a2 = lam58*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    d_a3 = lam18*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    d_a = (d_a1 + d_a2 + d_a3)/3
    print(d_a1, d_a2, d_a3)

    d_j1 = lam4*p/(lam1*M1+lam3*M3+lam4*M4)
    d_j2 = lam45*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    d_j3 = lam14*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    d_j4 = lam48*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    d_j = (d_j1 + d_j2 + d_j3 + d_j4)/4
    print(d_j1, d_j2, d_j3, d_j4)

    g_a1 = -s/M6 #
    g_a2 = -lam56*r*s/(a_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))
    g_a3 = -lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    g_a4 = -lam68*r*s/(d_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)) #
    g_a5 = -lam46*p*s/(d_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    g_a = (g_a1 + g_a2 + g_a3 + g_a4 + g_a5)/5
    print(g_a1, g_a2, g_a3, g_a4, g_a5)

    g_j1 = -q/M2 #
    g_j2 = -lam25*q*r/(a_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_j3 = -lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    g_j4 = -lam28*q*r/(d_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_j5 = -lam24*p*q/(d_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)) #
    g_j6 = q*s/(g_a*M2*M6)
    g_j = (g_j1 + g_j2 + g_j3 + g_j4 + g_j5 + g_j6)/6
    print(g_j1, g_j2, g_j3, g_j4, g_j5, g_j6)

    b_a1 = lam7*r/(lam5*M5+lam7*M7+lam8*M8)
    b_a2 = lam57*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    b_a3 = lam17*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_a4 = lam78*r*r/(2*d_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    b_a5 = lam47*p*r/(d_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_a6 = -lam67*r*s/(g_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))
    b_a7 = -lam27*q*r/(g_j*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    b_a = (b_a1 + b_a2 + b_a3 + b_a4 + b_a5 + b_a6 + b_a7)/7
    print(b_a1, b_a2, b_a3, b_a4, b_a5, b_a6, b_a7)

    b_j1 = lam3*p/(lam1*M1+lam3*M3+lam4*M4)
    b_j2 = lam35*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j3 = lam13*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    b_j4 = lam38*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j5 = lam34*p*p/(2*d_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    b_j6 = lam37*p*r/(b_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j7 = -lam36*p*s/(g_a*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    b_j8 = -lam23*p*q/(g_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    b_j = (b_j1 + b_j2 + b_j3 + b_j4 + b_j5 + b_j6 + b_j7 + b_j8)/8
    print(b_j1, b_j2, b_j3, b_j4, b_j5, b_j6, b_j7, b_j8)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_6(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8) = coefData.loc[lamsKey,
    ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8']]
    (lam11, lam12, lam13, lam14, lam15, lam16, lam17, lam18) = coefData.loc[lamsKey,
    ['lam11','lam12','lam13','lam14','lam15','lam16','lam17','lam18']]
    (lam22, lam23, lam24, lam25, lam26, lam27, lam28) = coefData.loc[lamsKey,
    ['lam22','lam23','lam24','lam25','lam26','lam27','lam28']]
    (lam33, lam34, lam35, lam36, lam37, lam38) = coefData.loc[lamsKey,
    ['lam33','lam34','lam35','lam36','lam37','lam38']]
    (lam44, lam45, lam46, lam47, lam48) = coefData.loc[lamsKey,
    ['lam44','lam45','lam46','lam47','lam48']]
    (lam55, lam56, lam57, lam58) = coefData.loc[lamsKey,
    ['lam55','lam56','lam57','lam58']]
    (lam66, lam67, lam68) = coefData.loc[lamsKey,
    ['lam66','lam67','lam68']]
    (lam77, lam78, lam88) = coefData.loc[lamsKey,
    ['lam77','lam78','lam88']]

    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)
    print(a_a)

    _a_j = []
    _a_j.append(lam1*p/(lam1*M1+lam3*M3+lam4*M4))
    _a_j.append(lam15*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _a_j = [elem for elem in _a_j if elem > 0]
    a_j = np.mean(_a_j)
    print(_a_j)

    _d_a = []
    _d_a.append(lam8*r/(lam5*M5+lam7*M7+lam8*M8))
    _d_a.append(lam58*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _d_a.append(lam18*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _d_a = [elem for elem in _d_a if elem > 0]
    d_a = np.mean(_d_a)
    print(_d_a)

    _d_j = []
    _d_j.append(lam4*p/(lam1*M1+lam3*M3+lam4*M4))
    _d_j.append(lam45*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _d_j.append(lam14*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _d_j.append(lam48*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _d_j = [elem for elem in _d_j if elem > 0]
    d_j = np.mean(_d_j)
    print(_d_j)

    _b_a = []
    _b_a.append(lam7*r/(lam5*M5+lam7*M7+lam8*M8))
    _b_a.append(lam57*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _b_a.append(lam17*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_a.append(lam78*r*r/(2*d_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8)))
    _b_a.append(lam47*p*r/(d_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_a = [elem for elem in _b_a if elem > 0]
    b_a = np.mean(_b_a)
    print(_b_a)

    _b_j = []
    _b_j.append(lam3*p/(lam1*M1+lam3*M3+lam4*M4))
    _b_j.append(lam35*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_j.append(lam13*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _b_j.append(lam38*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_j.append(lam34*p*p/(2*d_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4)))
    _b_j.append(lam37*p*r/(b_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8)))
    _b_j = [elem for elem in _b_j if elem > 0]
    b_j = np.mean(_b_j)
    print(_b_j)

    _g_a = []
    _g_a.append(-s/M6) #
    _g_a.append(-lam56*r*s/(a_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)))
    _g_a.append(-lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    _g_a.append(-lam68*r*s/(d_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))) #
    _g_a.append(-lam46*p*s/(d_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    _g_a.append(-lam67*r*s/(b_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)))
    _g_a.append(-lam36*p*s/(b_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6)))
    _g_a = [elem for elem in _g_a if elem > 0]
    g_a = np.mean(_g_a)
    print(_g_a)

    _g_j = []
    _g_j.append(-q/M2) #
    _g_j.append(-lam25*q*r/(a_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)))
    _g_j.append(-lam28*q*r/(d_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam24*p*q/(d_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))) #
    _g_j.append(-lam27*q*r/(b_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8)))
    _g_j.append(-lam23*p*q/(b_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)))
    _g_j.append(q*s/(g_a*M2*M6))
    _g_j = [elem for elem in _g_j if elem > 0]
    g_j = np.mean(_g_j)
    print(_g_j)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def factParam(p, q, r, s, coefData, mpData, optPntId, lamsKey):
    """return order: a_j, g_j, b_j, d_j, a_a, g_a, b_a, d_a"""
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8) = coefData.loc[lamsKey,
    ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8']]
    (lam11, lam12, lam13, lam14, lam15, lam16, lam17, lam18) = coefData.loc[lamsKey,
    ['lam11','lam12','lam13','lam14','lam15','lam16','lam17','lam18']]
    (lam22, lam23, lam24, lam25, lam26, lam27, lam28) = coefData.loc[lamsKey,
    ['lam22','lam23','lam24','lam25','lam26','lam27','lam28']]
    (lam33, lam34, lam35, lam36, lam37, lam38) = coefData.loc[lamsKey,
    ['lam33','lam34','lam35','lam36','lam37','lam38']]
    (lam44, lam45, lam46, lam47, lam48) = coefData.loc[lamsKey,
    ['lam44','lam45','lam46','lam47','lam48']]
    (lam55, lam56, lam57, lam58) = coefData.loc[lamsKey,
    ['lam55','lam56','lam57','lam58']]
    (lam66, lam67, lam68) = coefData.loc[lamsKey,
    ['lam66','lam67','lam68']]
    (lam77, lam78, lam88) = coefData.loc[lamsKey,
    ['lam77','lam78','lam88']]
    lam = coefData.loc[lamsKey].values

    def formulas_0(curPar):
        if (curPar == 0 or curPar == 2 or curPar == 3):
            return lam[curPar]*p/(lam1*M1+lam3*M3+lam4*M4)
        if (curPar == 4 or curPar == 6 or curPar == 7):
            return lam[curPar]*r/(lam5*M5+lam7*M7+lam8*M8)
        if (curPar == 1):
            return -q/M2
        if (curPar == 5):
            return -s/M6

    def formulas(res, prevPar, curPar):
        calcLam = 7
        if (prevPar < curPar):
            fst = prevPar
            sec = curPar
        else:
            fst = curPar
            sec = prevPar
        for i in range(fst):
            for j in range(i, 8):
                calcLam += 1
        for j in range(fst, sec+1):
            calcLam += 1

        if ((curPar == 0 or curPar == 2 or curPar == 3) and (prevPar == 0 or prevPar == 2 or prevPar == 3)):
            return lam[calcLam]*p*p/(2*res[prevPar]*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3
                                                                                        + lam14*M1*M4 + lam34*M3*M4))
        if ((curPar == 4 or curPar == 6 or curPar == 7) and (prevPar == 4 or prevPar == 6 or prevPar == 7)):
            return lam[calcLam]*r*r/(2*res[prevPar]*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7
                                                                                        + lam58*M5*M8 + lam78*M7*M8))
        if  (((curPar == 0 or curPar == 2 or curPar == 3) and (prevPar == 4 or prevPar == 6 or prevPar == 7))
        or ((curPar == 4 or curPar == 6 or curPar == 7) and (prevPar == 0 or prevPar == 2 or prevPar == 3))):
            return lam[calcLam]*p*r/(res[prevPar]*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7
                                                            + lam38*M3*M8 + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
        if (((curPar == 0 or curPar == 2 or curPar == 3) and prevPar == 1)
        or ((prevPar == 0 or prevPar == 2 or prevPar == 3) and curPar == 1)):
            return -lam[calcLam]*p*q/(res[prevPar]*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
        if (((curPar == 0 or curPar == 2 or curPar == 3) and prevPar == 5)
        or ((prevPar == 0 or prevPar == 2 or prevPar == 3) and curPar == 5)):
            return -lam[calcLam]*p*s/(res[prevPar]*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
        if (((curPar == 4 or curPar == 6 or curPar == 7) and prevPar == 1)
        or ((prevPar == 4 or prevPar == 6 or prevPar == 7) and curPar == 1)):
            return -lam[calcLam]*q*r/(res[prevPar]*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
        if (((curPar == 4 or curPar == 6 or curPar == 7) and prevPar == 5)
        or ((prevPar == 4 or prevPar == 6 or prevPar == 7) and curPar == 5)):
            return -lam[calcLam]*r*s/(res[prevPar]*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))
        if ((curPar == 1 and prevPar == 5) or (curPar == 5 and prevPar == 1)):
            return q*s/(res[prevPar]*M2*M6)

    def calcRes(res, prevPars, curPar):
        tmp = []
        tmp.append(formulas_0(curPar))
        for prevPar in prevPars:
            tmp.append(formulas(res, prevPar, curPar))
        return np.mean(tmp)

    params = [0, 1, 2, 3, 4, 5, 6, 7]  # a_j, g_j, b_j, d_j, a_a, g_a, b_a, d_a
    resList = []
    for par1 in range(8):
        res = np.empty(9, dtype=object)  # dtype=object is less precise than dtype=float64, but need for element-list
        prevPars = []
        res[par1] = calcRes(res, prevPars, par1)
        prevPars.append(par1)
        _pars2 = copy.deepcopy(params)
        _pars2.remove(par1)
        for par2 in _pars2:
            res[par2] = calcRes(res, prevPars, par2)
            prevPars.append(par2)
            _pars3 = copy.deepcopy(_pars2)
            _pars3.remove(par2)
            for par3 in _pars3:
                res[par3] = calcRes(res, prevPars, par3)
                prevPars.append(par3)
                _pars4 = copy.deepcopy(_pars3)
                _pars4.remove(par3)
                for par4 in _pars4:
                    res[par4] = calcRes(res, prevPars, par4)
                    prevPars.append(par4)
                    _pars5 = copy.deepcopy(_pars4)
                    _pars5.remove(par4)
                    for par5 in _pars5:
                        res[par5] = calcRes(res, prevPars, par5)
                        prevPars.append(par5)
                        _pars6 = copy.deepcopy(_pars5)
                        _pars6.remove(par5)
                        for par6 in _pars6:
                            res[par6] = calcRes(res, prevPars, par6)
                            prevPars.append(par6)
                            _pars7 = copy.deepcopy(_pars6)
                            _pars7.remove(par6)
                            for par7 in _pars7:
                                res[par7] = calcRes(res, prevPars, par7)
                                prevPars.append(par7)
                                _pars8 = copy.deepcopy(_pars7)
                                _pars8.remove(par7)
                                for par8 in _pars8:  #par8 = _pars8[0]
                                    res[par8] = calcRes(res, prevPars, par8)
                                    prevPars.append(par8)
                                    res[8] = copy.deepcopy(prevPars)
                                    resList.append(np.copy(res))
                                    prevPars.pop()  # pop par8
                                prevPars.pop()  # pop par7
                            prevPars.pop()  # pop par6
                        prevPars.pop()  # pop par5
                    prevPars.pop()  # pop par4
                prevPars.pop()  # pop par3
            prevPars.pop()  # pop par2
    result = np.array(resList)
    return result

def restoreParam_7(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    result = factParam(p, q, r, s, coefData, mpData, optPntId, lamsKey)
    g_j = np.copy(result[:, 1])
    g_a = np.copy(result[:, 5])
    b_j = result[:, 2]
    b_a = result[:, 6]
    result[:, 1] = b_j
    result[:, 5] = b_a
    result[:, 2] = g_j
    result[:, 6] = g_a

    resParam = [result[:, j].mean() for j in range(8)]
    print(resParam)
    return resParam

def calcResData(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    result = factParam(p, q, r, s, coefData, mpData, optPntId, lamsKey)
    resData = pd.DataFrame(result, columns=['a_j', 'g_j', 'b_j', 'd_j', 'a_a', 'g_a', 'b_a', 'd_a', 'order'])
    print(resData)
    return resData

def checkParam(stratData, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    def sigHandler(signum, frame):
        raise TimeoutError("time is over")
    signal.signal(signal.SIGALRM, sigHandler)

    resData = calcResData(p, q, r, s, coefData, mpData, optPntId, lamsKey)
    result = resData.loc[:, :'d_a'].values

    odeRes = np.empty(len(result))
    for i in tqdm.tqdm(range(len(result))):
        a_j, g_j, b_j, d_j, a_a, g_a, b_a, d_a = result[i]
        print(a_j, g_j, b_j, d_j, a_a, g_a, b_a, d_a)
        if any(elem < 0 for elem in (a_j, g_j, b_j, d_j, a_a, g_a, b_a, d_a)):
            odeRes[i] = -1
            continue
        pqrsData = gs.calcPqrsData(mpData, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
        try:
            signal.alarm(5)
            rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=5001, z0=0.001, F0=0.001, _method='BDF')
            signal.alarm(0)
            popData, FLim = gs.analyzePopDynamics(stratData, rawPopData, eps=0.01)
            res = popData['t'].idxmax()
            odeRes[i] = res if (popData.loc[res, 't'] == 5000) else -3
        except ValueError as exc:
            print(exc)
            odeRes[i] = -2
        except TimeoutError as exc:
            print(exc)
            odeRes[i] = -4

    resData.loc[:, 'odeRes'] = odeRes
    return resData

def restoreParam_k1(FLim, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    """под restorePQRS_1"""
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[lamsKey, 'lam1':'lam8']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam66 = coefData.loc[lamsKey, 'lam66']

    hq = -2/(1 - lam6*lam26/(2*lam2*lam66))
    k = lam2*M2/(hq*FLim*q)
    print("k_q:", k)
    g_j = -lam2/(FLim*k*hq)  # substitute k --> the same: g_j = -q/M2

    hs = -2 - hq  # the same: hs = lam6*lam26/(lam2*lam66*(1 - lam6*lam26/(2*lam2*lam66)))
    k_s = lam6*M6/(hs*FLim*s)
    print("k_s:", k_s)  # k_s == k_q only if p,q,r,s = restorePQRS_1, otherwise k_s != k_q
    g_a = -lam6/(FLim*k_s*hs)  # substitute k_s --> the same: g_a = -s/M6

    hp = (lam1*M1+lam3*M3+lam4*M4)/(k*p)
    a_j = lam1/(k*hp)
    b_j = lam3/(k*hp)
    d_j = lam4/(k*hp)

    hr = (lam5*M5+lam7*M7+lam8*M8)/(k*r)
    a_a = lam5/(k*hr)
    b_a = lam7/(k*hr)
    d_a = lam8/(k*hr)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_k2(FLim, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1):
    """под restorePQRS_2"""
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[lamsKey, 'lam1':'lam8']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam22 = coefData.loc[lamsKey, 'lam22']

    hs = -2/(1 - lam2*lam26/(2*lam6*lam22))
    k = lam6*M6/(hs*FLim*s)
    print("k_s:", k)
    g_a = -lam6/(FLim*k*hs)  # substitute k --> the same: g_a = -s/M6

    hq = -2 - hs
    k_q = lam2*M2/(hq*FLim*q)
    print("k_q:", k_q)  # k_q == k_s only if p,q,r,s = restorePQRS_2, otherwise k_q != k_s
    g_j = -lam2/(FLim*k_q*hq)  # substitute k_q --> the same: g_j = -q/M2

    hp = (lam1*M1+lam3*M3+lam4*M4)/(k*p)
    a_j = lam1/(k*hp)
    b_j = lam3/(k*hp)
    d_j = lam4/(k*hp)

    hr = (lam5*M5+lam7*M7+lam8*M8)/(k*r)
    a_a = lam5/(k*hr)
    b_a = lam7/(k*hr)
    d_a = lam8/(k*hr)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_all_k(FLim, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1, qs_key='qs'):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[lamsKey, 'lam1':'lam8']
    lam26 = coefData.loc[lamsKey, 'lam26']
    lam22 = coefData.loc[lamsKey, 'lam22']
    lam66 = coefData.loc[lamsKey, 'lam66']

    hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = tr.calcDers_qFsF(p, q, r, s, FLim)

    k_q = (lam2*M2)/(hq*FLim*q)
    k_s = (lam6*M6)/(hs*FLim*s)
    k_ss = (2*lam66*M6*M6)/(hss*FLim*FLim*s*s)
    k_qq = (2*lam22*M2*M2)/(hqq*FLim*FLim*q*q)
    k_qs = (lam26*M2*M6)/(hqs*FLim*FLim*q*s)
    print("k_q:", k_q)
    print("k_s:", k_s)
    print("k_ss:", k_ss)
    print("k_qq:", k_qq)
    print("k_qs:", k_qs)

    if (qs_key == 'q'):
        k = k_q
    elif (qs_key == 's'):
        k = k_s
    elif (qs_key == 'qs'):
        k = k_qs  # если restorePQRS_1, то k_qs == kss , если restorePQRS_2, то k_qs == kqq
    elif (qs_key == 'mean'):
        k = np.mean([k_q, k_s, k_ss, k_qq, k_qs])
    else:
        return None
    print("--------------")
    print("k:", k)

    a_j = lam1/(k*hp)
    b_j = lam3/(k*hp)
    d_j = lam4/(k*hp)
    g_j = -lam2/(FLim*k*hq)

    a_a = lam5/(k*hr)
    b_a = lam7/(k*hr)
    d_a = lam8/(k*hr)
    g_a = -lam6/(FLim*k*hs)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

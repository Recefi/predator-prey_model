import pandas as pd
import libs.param as param

def restorePQRS(FLim, stratPopData, coefData, mpData, optPntId):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    M2 = mpData.loc[optPntId, 'M2']
    M6 = mpData.loc[optPntId, 'M6']
    lam26 = coefData.loc[-1, 'lam26']
    lam66 = coefData.loc[-1, 'lam66']
    
    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim
    q = (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z1Lim-(lam66*M6*z2Lim)/(lam26*M2)))
    s = -(lam66*M6*q)/(lam26*M2)
    p = (-q*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim

    return p, q, r, s

def restorePQRS_2(FLim, stratPopData, coefData, mpData, optPntId):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    M2 = mpData.loc[optPntId, 'M2']
    M6 = mpData.loc[optPntId, 'M6']
    lam26 = coefData.loc[-1, 'lam26']
    lam22 = coefData.loc[-1, 'lam22']
    
    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim
    s = (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z2Lim-(lam22*M2*z1Lim)/(lam26*M6)))
    q = -(lam22*M2*s)/(lam26*M6)
    p = (-q*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim

    return p, q, r, s

def restoreParam(p, q, r, s, coefData, mpData, optPntId):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[-1, 'lam1':'lam8']

    g_j = -q/M2
    g_a = -s/M6

    # h2 = -2*FLim/(1-lam6*lam26/(lam2*lam66))
    # k = lam2*M2/(h2*FLim*q)
    # h1 = (lam1*M1+lam3*M3+lam4*M4)/(k*p)
    # h3 = (lam5*M5+lam7*M7+lam8*M8)/(k*r)

    a_j = lam1*p/(lam1*M1+lam3*M3+lam4*M4)
    b_j = lam3*p/(lam1*M1+lam3*M3+lam4*M4)
    d_j = lam4*p/(lam1*M1+lam3*M3+lam4*M4)

    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)
    b_a = lam7*r/(lam5*M5+lam7*M7+lam8*M8)
    d_a = lam8*r/(lam5*M5+lam7*M7+lam8*M8)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_2(p, q, r, s, coefData, mpData, optPntId):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1,lam3,lam4,lam11,lam12,lam13,lam14,lam15,lam16,lam17,lam18,lam23,lam24,lam33,lam34,lam35,lam36,lam37,lam38,
    lam44,lam45,lam46,lam47,lam48) = coefData.loc[-1, ['lam1','lam3','lam4','lam11','lam12','lam13','lam14','lam15',
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

    g_j = lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    g_a = lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_3(p, q, r, s, coefData, mpData, optPntId):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam5,lam7,lam8,lam13,lam15,lam17,lam18,lam25,lam27,lam28,lam35,lam37,lam38,lam45,lam47,lam48,
    lam55,lam56,lam57,lam58,lam67,lam68,lam77,lam78,lam88) = coefData.loc[-1, ['lam5','lam7','lam8','lam13','lam15',
    'lam17','lam18','lam25','lam27','lam28','lam35','lam37','lam38','lam45','lam47','lam48',
    'lam55','lam56','lam57','lam58','lam67','lam68','lam77','lam78','lam88']]


    a_a = lam5*r/(lam5*M5+lam7*M7+lam8*M8)

    b_a = lam57*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    d_a = lam58*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))

    a_j = lam15*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j = lam35*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    d_j = lam45*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))

    g_j = lam25*q*r/(a_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_a = lam56*r*s/(a_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

def restoreParam_4(p, q, r, s, coefData, mpData, optPntId):
    M1, M2, M3, M4, M5, M6, M7, M8 = mpData.loc[optPntId, 'M1':'M8']
    (lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8) = coefData.loc[-1,
    ['lam1','lam2','lam3','lam4','lam5','lam6','lam7','lam8']]
    (lam11, lam12, lam13, lam14, lam15, lam16, lam17, lam18) = coefData.loc[-1,
    ['lam11','lam12','lam13','lam14','lam15','lam16','lam17','lam18']]
    (lam22, lam23, lam24, lam25, lam26, lam27, lam28) = coefData.loc[-1,
    ['lam22','lam23','lam24','lam25','lam26','lam27','lam28']]
    (lam33, lam34, lam35, lam36, lam37, lam38) = coefData.loc[-1,
    ['lam33','lam34','lam35','lam36','lam37','lam38']]
    (lam44, lam45, lam46, lam47, lam48) = coefData.loc[-1,
    ['lam44','lam45','lam46','lam47','lam48']]
    (lam55, lam56, lam57, lam58) = coefData.loc[-1,
    ['lam55','lam56','lam57','lam58']]
    (lam66, lam67, lam68) = coefData.loc[-1,
    ['lam66','lam67','lam68']]
    (lam77, lam78, lam88) = coefData.loc[-1,
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

    b_a1 = lam7*r/(lam5*M5+lam7*M7+lam8*M8)
    b_a2 = lam57*r*r/(2*a_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    b_a3 = lam17*p*r/(a_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_a4 = lam78*r*r/(2*d_a*(lam55*M5**2 + lam77*M7**2 + lam88*M8**2 + lam57*M5*M7 + lam58*M5*M8 + lam78*M7*M8))
    b_a5 = lam47*p*r/(d_j*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_a = (b_a1 + b_a2 + b_a3 + b_a4 + b_a5)/5
    print(b_a1, b_a2, b_a3, b_a4, b_a5)

    b_j1 = lam3*p/(lam1*M1+lam3*M3+lam4*M4)
    b_j2 = lam35*p*r/(a_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j3 = lam13*p*p/(2*a_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    b_j4 = lam38*p*r/(d_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j5 = lam34*p*p/(2*d_j*(lam11*M1**2 + lam33*M3**2 + lam44*M4**2 + lam13*M1*M3 + lam14*M1*M4 + lam34*M3*M4))
    b_j6 = lam37*p*r/(b_a*(lam15*M1*M5 + lam17*M1*M7 + lam18*M1*M8 + lam35*M3*M5 + lam37*M3*M7 + lam38*M3*M8
                                                                        + lam45*M4*M5 + lam47*M4*M7 + lam48*M4*M8))
    b_j = (b_j1 + b_j2 + b_j3 + b_j4 + b_j5 + b_j6)/6
    print(b_j1, b_j2, b_j3, b_j4, b_j5, b_j6)

    g_a1 = -s/M6 #
    g_a2 = lam56*r*s/(a_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))
    g_a3 = lam16*p*s/(a_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    g_a4 = lam68*r*s/(d_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8)) #
    g_a5 = lam46*p*s/(d_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    g_a6 = lam67*r*s/(b_a*(lam56*M5*M6 + lam67*M6*M7 + lam68*M6*M8))
    g_a7 = lam36*p*s/(b_j*(lam16*M1*M6 + lam36*M3*M6 + lam46*M4*M6))
    g_a = (g_a1 + g_a2 + g_a3 + g_a4 + g_a5 + g_a6 + g_a7)/7
    print(g_a1, g_a2, g_a3, g_a4, g_a5, g_a6, g_a7)

    g_j1 = -q/M2 #
    g_j2 = lam25*q*r/(a_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_j3 = lam12*p*q/(a_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    g_j4 = lam28*q*r/(d_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_j5 = lam24*p*q/(d_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4)) #
    g_j6 = lam27*q*r/(b_a*(lam25*M2*M5 + lam27*M2*M7 + lam28*M2*M8))
    g_j7 = lam23*p*q/(b_j*(lam12*M1*M2 + lam23*M2*M3 + lam24*M2*M4))
    g_j8 = q*s/(g_a*M2*M6)
    g_j = (g_j1 + g_j2 + g_j3 + g_j4 + g_j5 + g_j6 + g_j7 + g_j8)/8
    print(g_j1, g_j2, g_j3, g_j4, g_j5, g_j6, g_j7, g_j8)

    return a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a

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

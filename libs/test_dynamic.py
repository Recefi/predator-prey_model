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

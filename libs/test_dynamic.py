import pandas as pd
import libs.param as param

def restorePQRS(FLim, stratPopData, coefData, mpData, optPntId):
    z1Lim = stratPopData.loc[optPntId, 'z1']
    z2Lim = stratPopData.loc[optPntId, 'z2']
    mp2 = mpData.loc[optPntId, 'M2']
    mp6 = mpData.loc[optPntId, 'M6']
    lam26 = coefData.loc[-1, 'lam26']
    lam66 = coefData.loc[-1, 'lam66']
    

    r = (FLim + (z1Lim+z2Lim)**2)/z2Lim
    q = (r*z2Lim - (z1Lim + z2Lim)**2)/(FLim*(z1Lim-(lam66*mp6*z2Lim)/(lam26*mp2)))
    s = -(lam66*mp6*q)/(lam26*mp2)
    p = (-q*FLim*z1Lim + r*z2Lim - z1Lim*(z1Lim + z2Lim))/z1Lim

    return p, q, r, s

def restoreParam(p, q, r, s, coefData, mpData, optPntId):
    mp1, mp2, mp3, mp4, mp5, mp6, mp7, mp8 = mpData.loc[optPntId, 'M1':'M8']
    lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = coefData.loc[-1, 'lam1':'lam8']
    lam26, lam66 = coefData.loc[-1, ['lam26', 'lam66']]

    g_j = -q/mp2
    g_a = -s/mp6

    # h2 = -2*FLim/(1-lam6*lam26/(lam2*lam66))
    # k = lam2*Mp2/(h2*FLim*q)
    # h1 = (lam1*mp1+lam3*mp3+lam4*mp4)/(k*p)
    # h3 = (lam5*mp5+lam7*mp7+lam8*mp8)/(k*r)

    a_j = lam1*p/(lam1*mp1+lam3*mp3+lam4*mp4)
    b_j = lam3*p/(lam1*mp1+lam3*mp3+lam4*mp4)
    d_j = lam4*p/(lam1*mp1+lam3*mp3+lam4*mp4)

    a_a = lam5*r/(lam5*mp5+lam7*mp7+lam8*mp8)
    b_a = lam7*r/(lam5*mp5+lam7*mp7+lam8*mp8)
    d_a = lam8*r/(lam5*mp5+lam7*mp7+lam8*mp8)

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

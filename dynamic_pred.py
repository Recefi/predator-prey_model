import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess

import libs.gen_selection as gs
import libs.graphical_interface as gui
import libs.utility as ut
import libs.machine_learning as ml
import libs.test_result as tr
import libs.param as param


# stratData = gs.genStrats(500, "beta")
# stratData.loc[len(stratData.index)] = [-34.58, -3.29, -83.32, -51.57]
# ut.writeData(stratData, "strat_data")
stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")

start = time.time()
rawPopData = gs.calcPopDynamics(pqrsData, tMax=2000, tParts=20000, z0=0.001, F0=0.1)
print ("calc pop dynamics: ", time.time() - start)
# start = time.time()
# ut.writeData(rawPopData, "raw_pop_data")
# print ("write pop dynamics: ", time.time() - start)

gui.popDynamics(rawPopData)
gui.corrMps(mpData)
#plt.show()

start = time.time()
stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
print ("analyze pop dynamics: ", time.time() - start)
ut.writeData(stratPopData, "strat_pop_data")
mpData = mpData.loc[stratPopData.index]
print("strats: ", len(stratPopData.index))

start = time.time()
selData = gs.calcSelection(stratPopData, mpData)
print ("calc sel time: ", time.time() - start)
start = time.time()
ut.writeData(selData, "sel_data")
print ("write sel time: ", time.time() - start)

#gui.histMps(selData)
# plt.show()


norm_mlLams, mpMaxsData = ml.runClfSVM(selData)
ut.writeData(mpMaxsData, 'mp_maxs_data')
mpMaxs = mpMaxsData.values[0]
mlLams = tr.denormMlLams(norm_mlLams, mpMaxs)

norm_selData = selData.copy()
norm_selData.loc[:,'M1':'M8M8'] = norm_selData.loc[:,'M1':'M8M8'] / mpMaxs

coefData = tr.getCoefData(pqrsData, norm_mlLams[1:], mlLams[1:], FLim)
ut.writeData(coefData, "coef_data")

#subprocess.Popen("python clfPlanes.py dynamic_pred --lam0="+str(norm_mlLams[0])+" --show", shell=True)


cosines = tr.getCosinesCoef(coefData)
print(cosines)
nearPntId = cosines.idxmax()
print(nearPntId)
print("nearPnt cosine:", cosines[nearPntId])
optPntId = stratPopData[['z1','z2']].sum(axis="columns").idxmax()
print(optPntId)
print("optPnt cosine:", cosines[optPntId], "\n")

compareCoefData = tr.compareCoefs(coefData, nearPntId, optPntId)
with pd.option_context('display.max_rows', 10):
    print(compareCoefData)


# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M2', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M6', 'M8', 25, -130)

#gui.clf2dPlane(norm_selData, norm_mlLams, 'M2', 'M4M8')

#plt.show()

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

p, q, r, s = restorePQRS(FLim, stratPopData, coefData, mpData, optPntId)

comparePqrsData = pd.DataFrame(columns=['p','q','r','s'])
comparePqrsData.loc['true'] = pqrsData.loc[optPntId, ['p','q','r','s']]
comparePqrsData.loc['restore'] = [p, q, r, s]
print(comparePqrsData)

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

a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = restoreParam(p, q, r, s, coefData, mpData, optPntId)

compareParamData = pd.DataFrame(columns=['a_j','b_j','g_j','d_j','a_a','b_a','g_a','d_a'])
compareParamData.loc['true'] = [param.alpha_j, param.beta_j, param.gamma_j, param.delta_j,
                                    param.alpha_a, param.beta_a, param.gamma_a, param.delta_a]
compareParamData.loc['restore'] = [a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a]
print(compareParamData)

_p, _q, _r, _s = pqrsData.loc[optPntId, ['p','q','r','s']]
_FLim = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
print(_FLim)
_FLim = gs.calcFLim(_p, _q, _r, _s, F0=100000)
print(_FLim)
_FLim = gs.calcFLim(_p, _q, _r, _s, F0=0.00001)
print(_FLim)

gui.stratSinsById(stratData, optPntId)
# plt.show()

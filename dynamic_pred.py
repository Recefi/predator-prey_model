import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess

import libs.gen_selection as gs
import libs.graphical_interface as gui
import libs.utility as ut
import libs.machine_learning as ml
import libs.taylor as tr
import libs.test_dynamic as td
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


p, q, r, s = td.restorePQRS(FLim, stratPopData, coefData, mpData, optPntId)
comparePqrsData = td.compareRestoredPQRS(p, q, r, s, pqrsData, optPntId)
print(comparePqrsData)

a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = td.restoreParam(p, q, r, s, coefData, mpData, optPntId)
compareParamData = td.compareRestoredParam(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
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

minsData = gs.fitBySel(pqrsData)
ut.writeData(minsData, "mins_data")

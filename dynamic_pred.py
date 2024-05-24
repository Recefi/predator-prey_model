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


# stratData = gs.genStrats(100, "beta", ab=5)
# #stratData = gs.genStrats(500, "uniform")
# #gs.filterStratsByBa(stratData, eps=0.25)
# stratData.loc[len(stratData.index) - 1] = [-34.58, -3.29, -83.32, -51.57]
# ut.writeData(stratData, "strat_data")
stratData = ut.readData("strat_data")

#gui.histStrats(stratData)
#plt.show()

mpData = gs.calcMpData(stratData)
ut.writeData(mpData, "mp_data")
pqrsData = gs.calcPqrsData(mpData)
ut.writeData(pqrsData, "pqrs_data")

start = time.time()
rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=100000, z0=0.001, F0=0.001)
print ("calc pop dynamics: ", time.time() - start)

start = time.time()
stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
print ("analyze pop dynamics: ", time.time() - start)
ut.writeData(stratPopData, "strat_pop_data")
shortMpData = mpData.loc[stratPopData.index]
print("strats: ", len(stratPopData.index))

#gui.popDynamics(rawPopData)
#gui.corrMps(shortMpData)
#gui.histStrats(stratData)
#plt.show()

start = time.time()
selData = gs.calcSelection(stratPopData, shortMpData, callerName="dynamic_pred")
print ("calc sel time: ", time.time() - start)
start = time.time()
ut.writeData(selData, "sel_data")
print ("write sel time: ", time.time() - start)

# gui.histMps(selData)
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


p, q, r, s = td.restorePQRS_2(FLim, stratPopData, coefData, mpData, optPntId)
comparePqrsData = td.compareRestoredPQRS(p, q, r, s, pqrsData, optPntId)
print(comparePqrsData)

a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = td.restoreParam(p, q, r, s, coefData, mpData, optPntId)
compareParamData = td.compareRestoredParam(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
print(compareParamData)
ut.writeData(compareParamData, "compare_param_data")

# _compareParamData = ut.readData("_compare_param_data")
# _compareParamData = pd.concat([_compareParamData, compareParamData], axis=0)
# ut.writeData(_compareParamData, "_compare_param_data")

_p, _q, _r, _s = pqrsData.loc[optPntId, ['p','q','r','s']]
_FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
print(_FLim)
z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
print(z1, z2)
gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
print()

_FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=1000)
print(_FLim)
z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
print(z1, z2)
gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
print()

_FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=-1000)
print(_FLim)
z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
print(z1, z2)
gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
print()


# stratMinsData, idOptStrat = gs.fitMaxMin(stratData, pqrsData)
# ut.writeData(stratMinsData, "strat_mins_data")
# print(stratMinsData.loc[idOptStrat])
# gui.stratSinsById(stratData, idOptStrat)
# plt.show()

# rstdPqrsData = gs.calcPqrsData(mpData, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# ut.writeData(rstdPqrsData, "pqrs_rstd_data")
# stratMinsData, idOptStrat = gs.fitMaxMin(stratData, rstdPqrsData)
# ut.writeData(stratMinsData, "strat_mins_rstd_data")
# print(stratMinsData.loc[idOptStrat])
# gui.stratSinsById(stratData, idOptStrat)
# plt.show()

# gui.compareStratSinsById(stratData, optPntId, idOptStrat)
# plt.show()


# FsolsData = gs.chkFsolsOnSel(stratData, pqrsData, abs=False)
# ut.writeData(FsolsData, "Fsols_data", subDirsName="Fsols")
# absFsolsData = gs.chkFsolsOnSel(stratData, pqrsData)
# ut.writeData(absFsolsData, "Fsols_abs_data", subDirsName="Fsols")
# complexFsolsData = gs.chkComplexFsolsOnSel(stratData, pqrsData)
# ut.writeData(complexFsolsData, "Fsols_complex_data", subDirsName="Fsols")
# rstdFsolsData = gs.chkFsolsOnSel(stratData, rstdPqrsData, abs=False)
# ut.writeData(rstdFsolsData, "Fsols_rstd_data", subDirsName="Fsols")
# rstdAbsFsolsData = gs.chkFsolsOnSel(stratData, rstdPqrsData)
# ut.writeData(rstdAbsFsolsData, "Fsols_abs_rstd_data", subDirsName="Fsols")
# rstdComplexFsolsData = gs.chkComplexFsolsOnSel(stratData, rstdPqrsData)
# ut.writeData(rstdComplexFsolsData, "Fsols_complex_rstd_data", subDirsName="Fsols")

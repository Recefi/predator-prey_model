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
import libs.restore_dynamic as rd
import libs.param as param
import libs.research as rs


# stratData = gs.genStrats(100, "beta", ab=5)
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
rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=100000, z0=0.001, F0=0.001, _method='Radau')
# with pd.option_context('display.max_rows', None):
#     print(rawPopData)
# ut.writeData(rawPopData, "raw_pop_data")
# ut.writeData(rawPopData.loc['z1_v5'], "z1_v5")
# ut.writeData(rawPopData.loc['z2_v5'], "z2_v5")
print ("calc pop dynamics: ", time.time() - start)

start = time.time()
stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
print ("analyze pop dynamics: ", time.time() - start)
ut.writeData(stratPopData, "strat_pop_data")
shortMpData = mpData.loc[stratPopData.index]
print("strats: ", len(stratPopData.index))

# stratFitData = gs.calcStratFitData(stratData, pqrsData, F=FLim)
# ut.writeData(stratFitData, "strat_fit_data")
# shortMpData = mpData.loc[stratFitData.index]
# print("strats: ", len(stratFitData.index))

stratPopFitData = gs.calcStratFitData(stratPopData, pqrsData.loc[stratPopData.index], F=0.3383)
ut.writeData(stratPopFitData, "strat_pop_fit_data")
print("checkRanking: ", gs.checkRanking(stratPopFitData))

#gui.popDynamics(rawPopData)
#gui.corrMps_2(shortMpData)
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


print("checkMl: ", gs.checkMl(selData, coefData, idx=-1))
print("checkTaylor: ", gs.checkMl(selData, coefData, idx=99))

popMlData = gs.calcPopLinsumData(stratPopData, mpData, coefData, idx=-1)
ut.writeData(popMlData, 'pop_ml_data')
print("checkMlRanking: ", gs.checkRanking(popMlData))
popTaylorData = gs.calcPopLinsumData(stratPopData, mpData, coefData, idx=99)
ut.writeData(popTaylorData, 'pop_taylor_data')
print("checkTaylorRanking: ", gs.checkRanking(popTaylorData))


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
ut.writeData(compareCoefData, "compare_coef_data")


#gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M3', 'M4', 6, -36, a=0.2)
#gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130, a=0.2)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M2', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M6', 'M8', 25, -130)

#gui.clf2dPlane(norm_selData, norm_mlLams, 'M2', 'M4M8')

#gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M5', 'M2M6', 6, -36, a=0.2)  # 5, -35  # 6, -40  # 6, -36

#plt.show()


#
# p, q, r, s = pqrsData.loc[optPntId, ['p','q','r','s']]
# print(rd.qzsz_1(q, s, stratPopData, optPntId))
# print(rd.qzsz_2(r, FLim, stratPopData, optPntId))
#

p, q, r, s = rd.restorePQRS_2(FLim, stratPopData, coefData, mpData, optPntId, lamsKey=-1)
comparePqrsData = rd.compareRestoredPQRS(p, q, r, s, pqrsData, optPntId)
print(comparePqrsData)
ut.writeData(comparePqrsData, "compare_pqrs_data")

#a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = rd.restoreParam_4(p, q, r, s, coefData, mpData, optPntId, lamsKey=-1)
a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = rd.restoreParam_k2(FLim, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1)
compareParamData = rd.compareRestoredParam(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
print(compareParamData)
ut.writeData(compareParamData, "compare_param_data")

# checkParamData = rd.checkParam(stratData, p, q, r, s, coefData, mpData, optPntId, lamsKey=-1)
# ut.writeData(checkParamData, "check_param_data")

#
# p, q, r, s = rd.checkPqrs(mpData, optPntId, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# print(p, q, r, s)
#

#
# rstdPqrsData = gs.calcPqrsData(mpData, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# rawPopData = gs.calcPopDynamics(rstdPqrsData, tMax=5000, tParts=100000, z0=0.001, F0=0.001)
# stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
# print(stratPopData.loc[optPntId])
# print(gs.calcZLim(p, q, r, s, FLim))

# print(rd.qzsz_1(q, s, stratPopData, optPntId))
# print(rd.qzsz_2(r, FLim, stratPopData, optPntId))

# print()

# _compareParamData = ut.readData("_compare_param_data")
# _compareParamData = pd.concat([_compareParamData, compareParamData], axis=0)
# ut.writeData(_compareParamData, "_compare_param_data")
#

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


gui.mostOptStratSins(stratPopData,3,4, key='t', title="Ранжирование по динамике видов")

# stratFitData_linsum = gs.calcStratFitData_linsum(stratPopData, mpData.loc[stratPopData.index], coefData)
# gui.mostOptStratSins(stratFitData_linsum,3,4, title="Ранжирование по лин.свертке фитнеса с восст.лямбда")

# stratFitData_linsum = gs.calcStratFitData_linsum(stratPopData, mpData.loc[stratPopData.index], coefData, idx=optPntId)
# gui.mostOptStratSins(stratFitData_linsum,3,4, title="Ранжирование по Тейлору для оптимальной")

# stratFitData = gs.calcStratFitData(stratPopData, pqrsData.loc[stratPopData.index], F=FLim)
# gui.mostOptStratSins(stratFitData,3,4, title="Ранжирование по расчетной формуле фитнеса с исх.параметрами")

# rstdPqrsData = gs.calcPqrsData(mpData.loc[stratPopData.index], a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# stratFitData = gs.calcStratFitData(stratPopData, rstdPqrsData, F=FLim)
# gui.mostOptStratSins(stratFitData,3,4, title="Ранжирование по расчетной формуле фитнеса с восст.параметрами с исх.F*")

# stratMinsData, idOptStrat = gs.fitMaxMin(stratPopData, pqrsData.loc[stratPopData.index])
# gui.mostOptStratSins(stratMinsData,3,4, key='min', title="Ранжирование по максминной задаче с исх.параметрами")

# rstdPqrsData = gs.calcPqrsData(mpData.loc[stratPopData.index], a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
# rstdStratMinsData, idOptStrat = gs.fitMaxMin(stratPopData, rstdPqrsData)
# gui.mostOptStratSins(rstdStratMinsData,3,4, key='min', title="Ранжирование по максминной задаче с восст.параметрами")

rstdPqrsData = gs.calcPqrsData(mpData.loc[stratPopData.index], a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
rawPopData = gs.calcPopDynamics(rstdPqrsData, tMax=5000, tParts=100000, z0=0.001, F0=0.001, _method="Radau")
rstdStratPopData, FLim = gs.analyzePopDynamics(stratData.loc[stratPopData.index], rawPopData, 0.01)
gui.mostOptStratSins(rstdStratPopData,3,4, key='t', title="Ранжирование по динамике видов с восст.параметрами")

plt.show()


# pqrsRow = pqrsData.loc[[optPntId]]
# stratRow = stratData.loc[[optPntId]]
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=5000, tParts=100000, z0=0.001, F0=0.001)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=5000, tParts=100000, z0=0.001, F0=1000)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)

# compareSearchFsolsData = rs.compareSearchFsols(stratData, pqrsData)
# ut.writeData(compareSearchFsolsData, "compare_search_Fsols", subDirsName="Fsols")


# gui.phasePortrait(_p, _q, _r, _s)
# plt.show()


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


# #FsolsData = rs.chkFsolsOnSel(stratData, pqrsData, abs=False)
# #ut.writeData(FsolsData, "Fsols_data", subDirsName="Fsols")
# absFsolsData = rs.chkFsolsOnSel(stratData, pqrsData)
# ut.writeData(absFsolsData, "Fsols_abs_data", subDirsName="Fsols")
# #complexFsolsData = rs.chkComplexFsolsOnSel(stratData, pqrsData)
# #ut.writeData(complexFsolsData, "Fsols_complex_data", subDirsName="Fsols")
# #rstdFsolsData = rs.chkFsolsOnSel(stratData, rstdPqrsData, abs=False)
# #ut.writeData(rstdFsolsData, "Fsols_rstd_data", subDirsName="Fsols")
# #rstdAbsFsolsData = rs.chkFsolsOnSel(stratData, rstdPqrsData)
# #ut.writeData(rstdAbsFsolsData, "Fsols_abs_rstd_data", subDirsName="Fsols")
# #rstdComplexFsolsData = rs.chkComplexFsolsOnSel(stratData, rstdPqrsData)
# #ut.writeData(rstdComplexFsolsData, "Fsols_complex_rstd_data", subDirsName="Fsols")

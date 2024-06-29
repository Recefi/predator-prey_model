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


# stratData = gs.genStrats(500, "beta")
# stratData.loc[len(stratData.index) - 1] = [-34.58, -3.29, -83.32, -51.57]
# ut.writeData(stratData, "strat_data")
stratData = ut.readData("strat_data")
#stratData = ut.readData("strat_pop_data")

print(stratData)

FLim = 1

mpData = gs.calcMpData(stratData)
ut.writeData(mpData, "mp_data")
pqrsData = gs.calcPqrsData(mpData)
ut.writeData(pqrsData, "pqrs_data")

# gui.histStrats(stratData)
# mpData.plot.scatter(x='M2', y='M1', s=5)
# plt.show()

stratFitData = gs.calcStratFitData(stratData, pqrsData, F=FLim)
ut.writeData(stratFitData, "strat_fit_data")
shortMpData = mpData.loc[stratFitData.index]
print("strats: ", len(stratFitData.index))

# gui.allStratSins(stratFitData)
# gui.optStratSins(stratFitData)
# gui.mostOptStratSins(stratFitData, 3, 4)
# gui.corrMps_2(shortMpData)
# plt.show()

start = time.time()
selData = gs.calcSelection(stratFitData, shortMpData)
print ("calc sel time: ", time.time() - start)
start = time.time()
ut.writeData(selData, "sel_data")
print ("write sel time: ", time.time() - start)

#gui.histMps(selData)
#plt.show()


norm_mlLams, mpMaxsData = ml.runClfSVM(selData)
ut.writeData(mpMaxsData, 'mp_maxs_data')
mpMaxs = mpMaxsData.values[0]
mlLams = tr.denormMlLams(norm_mlLams, mpMaxs)

norm_selData = selData.copy()
norm_selData.loc[:,'M1':'M8M8'] = norm_selData.loc[:,'M1':'M8M8'] / mpMaxs

coefData = tr.getCoefData(pqrsData, norm_mlLams[1:], mlLams[1:], F=FLim)
ut.writeData(coefData, "coef_data")

#subprocess.Popen("python clfPlanes.py static_pred --lam0="+str(norm_mlLams[0])+" --show", shell=True)


cosines = tr.getCosinesCoef(coefData)
print(cosines)
nearPntId = cosines.idxmax()
print(nearPntId)
print("nearPnt cosine:", cosines[nearPntId])
optPntId = stratFitData['fit'].idxmax()
print(optPntId)
print("optPnt cosine:", cosines[optPntId], "\n")

compareCoefData = tr.compareCoefs(coefData, nearPntId, optPntId)
with pd.option_context('display.max_rows', 10):
    print(compareCoefData)
ut.writeData(compareCoefData, "compare_coef_data")

compareFitsData, fitCosines = tr.compareFits(coefData, stratFitData, shortMpData, pqrsData, nearPntId, optPntId, F=FLim)
print(compareFitsData)
print(fitCosines)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M2', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M5', 'M6', 'M8', 25, -130)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M5', 'M4', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M5', 'M4M8')
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M2M6', 'M5', 25, -130)

# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M5', 'M1M5', 25, -130)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M1', 'M4', 'M1M4', 5, -25)
# gui.clf3dPlane(norm_selData, norm_mlLams, 'M6', 'M8', 'M6M8')

#gui.clf2dPlane(norm_selData, norm_mlLams, 'M2', 'M4M8')

plt.show()



# _p, _q, _r, _s = pqrsData.loc[optPntId, ['p','q','r','s']]
# _FLim = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
# print(_FLim)
# TODO: попробовать подогнать статическое F* к 1, мб результаты улучшаться...

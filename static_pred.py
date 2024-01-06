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


stratData = gs.genStrats(500, "beta")
ut.writeData(stratData, "strat_data")
# stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")

# gui.histStrats(stratData)
# mpData.plot.scatter(x='M2', y='M1', s=5)
# plt.show()

stratFitData = gs.calcFitness(stratData, pqrsData)
ut.writeData(stratFitData, "strat_fit_data")
mpData = mpData.loc[stratFitData.index]
print("strats: ", len(stratFitData.index))

# gui.allStratSins(stratData)
# gui.allStratSins(stratFitData)
# gui.optStratSin(stratFitData)
gui.mostOptStratSins(stratFitData, 3, 4)
gui.corrMps(mpData)
#plt.show()

start = time.time()
selData = gs.calcSelection(stratFitData, mpData)
print ("calc sel time: ", time.time() - start)
start = time.time()
ut.writeData(selData, "sel_data")
print ("write sel time: ", time.time() - start)

#gui.histMps(selData)
plt.show()


norm_mlLams, maxs = ml.runClfSVM(selData)
mlLams = tr.denormMlLams(norm_mlLams, maxs)

coefData = tr.getCoefData(pqrsData, norm_mlLams[1:], mlLams[1:])
ut.writeData(coefData, "coef_data")

cosines = tr.getCosinesCoef(coefData)
print(cosines)
nearPntId = cosines.idxmax()
print("nearPnt cosine:", cosines[nearPntId])
maxFitPntId = stratFitData['fit'].idxmax()
print("maxFitPnt cosine:", cosines[maxFitPntId], "\n")

compareCoefData, compareCoefCosines = tr.compareCoefs(coefData, nearPntId, maxFitPntId)
print(compareCoefData, compareCoefCosines, sep='\n')
compareFitData, compareFitCosines = tr.compareFits(coefData, stratFitData, mpData, pqrsData, maxFitPntId, nearPntId)
print(compareFitData, compareFitCosines, sep='\n')


subprocess.Popen("python clfPlanes.py static_pred --lam0="+str(norm_mlLams[0])+" --show", shell=True)

gui.clf3dPlane(selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
gui.clf3dPlane(selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130)

gui.clf3dPlane(selData, norm_mlLams, 'M1', 'M2', 'M4', 25, -130)
gui.clf3dPlane(selData, norm_mlLams, 'M5', 'M6', 'M8', 25, -130)

# gui.clf2dPlane(selData, norm_mlLams, 'M2', 'M4M8')

plt.show()


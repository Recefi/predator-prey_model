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


stratData = gs.genStrats(500)
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

# gui.allStratSins(stratData)
# gui.allStratSins(stratFitData)
# gui.optStratSin(stratFitData)
gui.mostOptStratSins(stratFitData, 3, 4)
gui.corrMps(mpData.loc[stratFitData.index])
plt.show()

start = time.time()
selData = gs.calcSelection(stratFitData, mpData.loc[stratFitData.index])
# ut.writeData(selData, "sel_data")
gs.normSelection(selData)
_selData = selData.loc[0:]
end = time.time()
print ("calc sel time: ", end - start)

start = time.time()
ut.writeData(selData, "norm_sel_data")
end = time.time()
print ("write sel time: ", end - start)

gui.histMps(_selData)
plt.show()

norm_mlLams = ml.runClfSVM(_selData)
mlLams = tr.denormMlLams(norm_mlLams, selData)

coefData = tr.getCoefData(pqrsData, norm_mlLams[1:], mlLams[1:])
ut.writeData(coefData, "coef_data")

subprocess.Popen("python clfPlanes.py static_pred --lam0="+str(norm_mlLams[0])+" --show", shell=True)

gui.clf3dPlane(_selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
gui.clf3dPlane(_selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130)

gui.clf3dPlane(_selData, norm_mlLams, 'M1', 'M2', 'M4', 25, -130)
gui.clf3dPlane(_selData, norm_mlLams, 'M5', 'M6', 'M8', 25, -130)

# gui.clf2dPlane(_selData, norm_mlLams, 'M2', 'M4M8')

plt.show()


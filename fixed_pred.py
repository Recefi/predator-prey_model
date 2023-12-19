import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess

import source.gen_selection as gs
import source.graphical_interface as gui
import source.utility as ut
import source.machine_learning as ml


stratData = gs.genStrats(10000)
ut.writeData(stratData, "strat_data")
# stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
mpData.plot.scatter(x='M2', y='M1', s=5)

stratData = gs.сlearSelection(stratData, 20, 4)
stratData = gs.сlearSelection(stratData, 50, 3)
stratData = gs.сlearSelection(stratData, 70, 2)

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")

mpData.plot.scatter(x='M2', y='M1', s=5)
# fig, ax = plt.subplots()
# ax.scatter(x=mpData['M2'].sub(mpData['M2'].mean()).div(mpData['M2'].std()), y=mpData['M1'].sub(mpData['M1'].mean()).div(mpData['M1'].std()), s=5)
# ax.set(xlabel='M2', ylabel='M1')
plt.show()

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

ut.writeData(pd.DataFrame({'ml': norm_mlLams}), "norm_coef_data")
subprocess.Popen("python clfPlanes.py fixed_pred --show", shell=True)

gui.clf3dPlaneMPL(_selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
gui.clf3dPlaneMPL(_selData, norm_mlLams, 'M5', 'M7', 'M8', 25, -130)

gui.clf3dPlaneMPL(_selData, norm_mlLams, 'M2', 'M7', 'M8', 25, -130)
gui.clf3dPlaneMPL(_selData, norm_mlLams, 'M2', 'M6', 'M4M8', 0, -45)

# gui.clf2dPlane(_selData, norm_mlLams, 'M2', 'M4M8')

plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import subprocess

import source.gen_selection as gs
import source.graphical_interface as gui
import source.utility as ut
import source.machine_learning as ml


# stratData = gs.genStrats(125)
# ut.writeData(stratData, "strat_data")
stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")

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
ut.writeData(selData, "sel_data")

norm_selData = gs.normSelection(selData)
ut.writeData(norm_selData, "norm_sel_data")
norm_selData = gs.stdSelection(selData)
ut.writeData(norm_selData, "std_sel_data")
_norm_selData = norm_selData.loc[0:]
end = time.time()
print ("sel time: ", end - start)

print(_norm_selData)

gui.histMps(_norm_selData)
plt.show()

norm_mlLams = ml.runClfSVM(_norm_selData)

ut.writeData(pd.DataFrame({'ml': norm_mlLams}), "norm_coef_data")
subprocess.Popen("python clfPlanes.py fixed_pred --show", shell=True)

gui.clf3dPlaneMPL(_norm_selData, norm_mlLams, 'M1', 'M3', 'M4', 25, -130)
gui.clf3dPlaneMPL(_norm_selData, norm_mlLams, 'M2', 'M6', 'M4M8', 0, -45)
gui.clf2dPlane(_norm_selData, norm_mlLams, 'M2', 'M4M8')
gui.clf2dPlane(_norm_selData, norm_mlLams, 'M6', 'M4M8')
gui.clf3dPlaneMPL(_norm_selData, norm_mlLams, 'M2', 'M6', 'M4', 0, -45)
gui.clf3dPlaneMPL(_norm_selData, norm_mlLams, 'M2', 'M6', 'M8', 0, -45)

gui.clf3dPlaneMPL(_norm_selData, norm_mlLams, 'M1', 'M5', 'M4M8', 0, -45)

plt.show()


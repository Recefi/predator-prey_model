import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import source.gen_selection as gs
import source.graphical_interface as gui
import source.utility as ut
import source.machine_learning as ml


stratData = gs.genStrats(125)
ut.writeData(stratData, "strat_data")
# stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")
# mpData = ut.readData("mp_data")
# pqrsData = ut.readData("pqrs_data")

stratFitData = gs.calcFitness(stratData, pqrsData)
ut.writeData(stratFitData, "strat_fit_data")

# gui.showCorrMps(mpData.loc[stratFitData.index])
# gui.showAllSins(stratData)
# gui.showAllSins(stratFitData)
# gui.showOptSin(stratFitData)
gui.showMostOptSins(stratFitData, 3, 4)

start = time.time()
selData = gs.calcSelection(stratFitData, mpData.loc[stratFitData.index])
ut.writeData(selData, "sel_data")

norm_selData, colMaxs = gs.normSelection(selData)
# norm_selData = gs.stdSelection(selData)
ut.writeData(norm_selData, "norm_sel_data")
end = time.time()
print ("sel time: ", end - start)

gui.showHistMps(norm_selData)

norm_mlLams, lam0 = ml.runClfSVM(norm_selData)

gui.drawClf2dPlane(norm_selData, norm_mlLams, lam0, 0, 2)
plt.show()
gui.showClfPlanes(norm_selData, norm_mlLams, lam0)

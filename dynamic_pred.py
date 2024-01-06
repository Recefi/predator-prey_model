import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import libs.gen_selection as gs
import libs.graphical_interface as gui
import libs.utility as ut


stratData = gs.genStrats(800, "beta")
ut.writeData(stratData, "strat_data")
# stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")

start = time.time()
rawPopData = gs.calcPopDynamics(pqrsData, 1000, 40000)
print ("calc pop dynamics: ", time.time() - start)
start = time.time()
ut.writeData(rawPopData, "raw_pop_data")
print ("write pop dynamics: ", time.time() - start)

gui.popDynamics(rawPopData)
gui.corrMps(mpData)
#plt.show()

stratPopData = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
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
plt.show()

import numpy as np
import pandas as pd

import source.gen_selection as gs
import source.graphical_interface as gui
import source.csv_data as cd



stratData = gs.genStrats(40)
cd.writeData(stratData, "strat_data")
# stratData = cd.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
cd.writeData(mpData, "mp_data")
cd.writeData(pqrsData, "pqrs_data")
# mpData = cd.readData("mp_data")
# pqrsData = cd.readData("pqrs_data")

gui.showCorrMps(mpData)
# gui.showAllSins(stratData.loc[mpData.index])

rawPopData = gs.calcPopDynamics(pqrsData)
cd.writeData(rawPopData, "raw_pop_data")
gui.showPopDynamics(rawPopData)

popData = gs.analyzePopDynamics(pqrsData.index, rawPopData, 10**(-10))
cd.writeData(popData, "pop_data")

selData = gs.calcSelection(mpData, popData)
cd.writeData(selData, "sel_data")

normSelData, colMaxs = gs.normSelection(selData)
cd.writeData(normSelData, "norm_sel_data")

#gui.showHist(normSelData)

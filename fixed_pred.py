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

stratFitData = gs.calcFitness(stratData, pqrsData)
cd.writeData(stratFitData, "strat_fit_data")

# gui.showCorrMps(mpData.loc[stratFitData.index])
# gui.showAllSins(stratData)
# gui.showAllSins(stratFitData)
gui.showOptSin(stratFitData)

selData = gs.calcSelection(stratFitData, mpData.loc[stratFitData.index])
cd.writeData(selData, "sel_data")

normSelData, colMaxs = gs.normSelection(selData)
cd.writeData(normSelData, "norm_sel_data")

gui.showHist(normSelData)

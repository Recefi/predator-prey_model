import numpy as np
import pandas as pd

import source.gen_selection as gs
import source.graphical_interface as gui
import source.csv_data as cd


stratData = gs.genStrats(200)
cd.writeData(stratData, "strat_data")
# stratData = cd.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
cd.writeData(mpData, "mp_data")
cd.writeData(pqrsData, "pqrs_data")
# mpData = cd.readData("mp_data")
# pqrsData = cd.readData("pqrs_data")

# gui.showCorrMps(mpData)

rawPopData = gs.calcPopDynamics(pqrsData)
cd.writeData(rawPopData, "raw_pop_data")
gui.showPopDynamics(rawPopData)

stratPopData = gs.analyzePopDynamics(stratData, rawPopData, 0)
cd.writeData(stratPopData, "strat_pop_data")

# selData = gs.calcSelection(stratPopData, mpData)
# cd.writeData(selData, "sel_data")

# norm_selData, colMaxs = gs.normSelection(selData)
# cd.writeData(norm_selData, "norm_sel_data")

# gui.showHistMps(norm_selData)

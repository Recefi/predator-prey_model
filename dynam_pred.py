import numpy as np
import pandas as pd

import source.gen_selection as gs
import source.graphical_interface as gui
import source.utility as ut


stratData = gs.genStrats(125)
ut.writeData(stratData, "strat_data")
# stratData = ut.readData("strat_data")

mpData, pqrsData = gs.calcMps(stratData)
ut.writeData(mpData, "mp_data")
ut.writeData(pqrsData, "pqrs_data")
# mpData = ut.readData("mp_data")
# pqrsData = ut.readData("pqrs_data")

# gui.showCorrMps(mpData)

rawPopData = gs.calcPopDynamics(pqrsData)
ut.writeData(rawPopData, "raw_pop_data")
gui.showPopDynamics(rawPopData)

stratPopData = gs.analyzePopDynamics(stratData, rawPopData, 0)
ut.writeData(stratPopData, "strat_pop_data")

# selData = gs.calcSelection(stratPopData, mpData)
# ut.writeData(selData, "sel_data")

# norm_selData, colMaxs = gs.normSelection(selData)
# ut.writeData(norm_selData, "norm_sel_data")

# gui.showHistMps(norm_selData)

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

# gui.corrMps(mpData)

rawPopData = gs.calcPopDynamics(pqrsData)
ut.writeData(rawPopData, "raw_pop_data")
gui.popDynamics(rawPopData)

stratPopData = gs.analyzePopDynamics(stratData, rawPopData)
ut.writeData(stratPopData, "strat_pop_data")

# selData = gs.calcSelection(stratPopData, mpData)
# ut.writeData(selData, "sel_data")

# norm_selData = gs.normSelection(selData)
# ut.writeData(norm_selData, "norm_sel_data")

# gui.histMps(norm_selData)

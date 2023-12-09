import numpy as np
import pandas as pd

import source.gen_selection as gs
import source.graphical_interface as gui
import source.csv_data as cd


# Либо
Aj, Bj, Aa, Ba = gs.genStrats(40)
stratData = cd.collectStratData(Aj, Bj, Aa, Ba)
cd.writeData(stratData, "strat_data")
# # Либо
# stratData = cd.readData("strat_data")
# Aj, Bj, Aa, Ba = cd.parseStratData(stratData)

# Либо
Mps, OrigIndxs, pqrsData = gs.calcMps(stratData)
mpData = cd.collectMpData(Mps, OrigIndxs)
cd.writeData(mpData, "mp_data")
cd.writeData(pqrsData, "pqrs_data")
# # Либо
# mpData = cd.readData("mp_data")
# pqrsData = cd.readData("pqrs_data")
# Mps, OrigIndxs = cd.parseMpData(mpData)

# gui.drawCorrellation(np.transpose(Mps)[:8],["M1","M2","M3","M4","M5","M6","M7","M8"])

# gui.showAllSins(stratData.loc[OrigIndxs])

rawPopData = gs.calcPopData(pqrsData)
cd.writeData(rawPopData, "raw_pop_data")
gui.showPopDynamics(rawPopData)

popData = gs.analyzePopData(pqrsData.index, rawPopData, 10**(-10))
cd.writeData(popData, "pop_data")

selData1 = gs.calcSelData1(mpData, popData)
print(selData1)

selData2 = gs.calcSelData2(mpData, popData)
print(selData2)
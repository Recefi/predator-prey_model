import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import libs.gen_selection as gs
import libs.graphical_interface as gui
import libs.utility as ut
import libs.machine_learning as ml
import libs.taylor as tr
import libs.test_dynamic as td
import libs.param as param
import libs.research as rs

# stratData = gs.genStrats(10, "beta", ab=5)
# stratData.loc[len(stratData.index) - 1] = [-35.9, -3.6, -85.1, -52.5]
# stratData.loc[len(stratData.index) - 3] = [-32.7, -4.1, -81.2, -53.8]
# ut.writeData(stratData, "strat_data")
stratData = ut.readData("strat_data")

mpData = gs.calcMpData(stratData)
ut.writeData(mpData, "mp_data")
pqrsData = gs.calcPqrsData(mpData)
ut.writeData(pqrsData, "pqrs_data")

stratMinsData, idOptStrat = gs.fitMaxMin(stratData, pqrsData)
print(stratMinsData)

# _p, _q, _r, _s = (pqrsData[col].values for col in pqrsData[['p','q','r','s']])
# for i in range(len(stratData.index)):
#     for j in range(len(stratData.index)):
#         if (i != j):
#             print(i, j, rs.findFsols_2(_p[i],_q[i],_r[i],_s[i],_p[j],_q[j],_r[j],_s[j], abs=True, errEps=1e-15))

# pqrsRow = pqrsData.loc[[7, 9]]
# stratRow = stratData.loc[[7, 9]]
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=10000, tParts=100000, z0=0.001, F0=0.001)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)
# print(FLim)
# gui.popDynamics(rawPopData)
# plt.show()

stratMinsData_2, idOptStrat_2 = gs.fitMaxMin_2(stratData, pqrsData)
with pd.option_context('display.max_rows', None):
    print(stratMinsData_2)
ut.writeData(stratMinsData_2, "strat_mins_2_data")

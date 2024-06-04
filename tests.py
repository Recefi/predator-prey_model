import matplotlib.pyplot as plt

import libs.graphical_interface as gui
import libs.gen_selection as gs
import libs.utility as ut
import libs.research as rs


# #gui.stratSins(-35, 3.93, -83, 49.32)
# #gui.stratSins(-34.58, 3.29, -83.32, 51.57)
# gui.stratSinsPoints(-34.58, 3.29, -83.32, 51.57)
# gui.stratSinsPoints_2(-34.58, -3.29, -83.32, -51.57)
# plt.show()


# #stratData = gs.genStrats(10000, "beta")
stratData = ut.readData("strat_data", "dynamic_pred")

# #stratData.loc[len(stratData.index)] = [-34.58, -3.29, -83.32, -51.57]

mpData = gs.calcMpData(stratData)
pqrsData = gs.calcPqrsData(mpData)

# stratFitData = gs.calcStratFitData(stratData, pqrsData, F=0.577304230127284)
# mpData = mpData.loc[stratFitData.index]
# print("strats: ", len(stratFitData.index))
# #ut.writeData(stratFitData, "strat_fit_data")

# gui.mostOptStratSins(stratFitData, 3, 4)
# plt.show()


# pqrsData = ut.readData("pqrs_data", "dynamic_pred")
# _p, _q, _r, _s = pqrsData.loc[19, ['p','q','r','s']]
# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=10000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=-10000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()


# rstdPqrsData = ut.readData("pqrs_rstd_data", "dynamic_pred")
# _p, _q, _r, _s = rstdPqrsData.loc[132, ['p','q','r','s']]
# #Fsols = rs.findFsols(_p, _q, _r, _s)
# Fsols = rs.findComplexFsols(_p, _q, _r, _s)
# print(Fsols)
# FLams, errs = rs.chkFsols(_p, _q, _r, _s, Fsols)
# print(FLams)


# # 3sel 5strat
# # 3sel 88strat
# i = 491
# pqrsData = ut.readData("pqrs_data", "dynamic_pred")
# _p, _q, _r, _s = pqrsData.loc[i, ['p','q','r','s']]
# #Fsols = rs.findFsols(_p, _q, _r, _s, abs=False)
# Fsols = rs.findFsols(_p, _q, _r, _s)
# #Fsols = rs.findComplexFsols(_p, _q, _r, _s)
# print(Fsols)
# FLams, errs = rs.chkFsols(_p, _q, _r, _s, Fsols)
# print(FLams)

# pqrsRow = pqrsData.loc[[i, 10, 12, 15, 98, 112]]
# stratRow = stratData.loc[[i, 10, 12, 15, 98, 112]]
# stratFitRow = gs.calcStratFitData(stratRow, pqrsRow, F = 1)
# gui.mostOptStratSins(stratFitRow, 2, 2)
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=500, tParts=100000, z0=0.001, F0=0.001)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)
# print(FLim)
# gui.popDynamics(rawPopData)
# plt.show()

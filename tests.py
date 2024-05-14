import matplotlib.pyplot as plt

import libs.graphical_interface as gui
import libs.gen_selection as gs
import libs.utility as ut


#gui.stratSins(-35, 3.93, -83, 49.32)
# gui.stratSins(-34.58, 3.29, -83.32, 51.57)
# plt.show()


# stratData = gs.genStrats(10000, "beta")
# stratData = ut.readData("strat_data", "dynamic_pred")

# #stratData.loc[len(stratData.index)] = [-34.58, -3.29, -83.32, -51.57]

# mpData = gs.calcMpData(stratData)
# pqrsData = gs.calcPqrsData(mpData)

# stratFitData = gs.calcStratFitData(stratData, pqrsData, F=0.82596775326931)
# mpData = mpData.loc[stratFitData.index]
# print("strats: ", len(stratFitData.index))

# gui.mostOptStratSins_static(stratFitData, 3, 4)
# plt.show()


# pqrsData = ut.readData("pqrs_data", "dynamic_pred")
# _p, _q, _r, _s = pqrsData.loc[19, ['p','q','r','s']]
# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.checkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=10000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.checkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=-10000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.checkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()


pqrsData = ut.readData("pqrs_data", "dynamic_pred")
_p, _q, _r, _s = pqrsData.loc[171, ['p','q','r','s']]
# gs.calcFLimSympy(_p, _q, _r, _s, 0.1)

# Fsols = gs.findFsols(_p, _q, _r, _s, -1000, 1000, 1)
Fsols = gs.findComplexFsols(_p, _q, _r, _s, -1000, 1000, 1)
print(Fsols)
FLams, errs = gs.checkFsols(_p, _q, _r, _s, Fsols)
print(FLams)

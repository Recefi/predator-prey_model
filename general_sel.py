import time
import gc
import pandas as pd
import matplotlib.pyplot as plt

import libs.gen_selection as gs
import libs.utility as ut
import libs.graphical_interface as gui

# Aj, Bj, Aa, Ba = gs.genGenlStrats()
# genlStratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
# ut.writeData(genlStratData, "general_strat_data")
# p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba)
# #pqrsData = pd.DataFrame({'p': p,'q': q,'r': r,'s': s})
# genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
# print(genlStratMinsData.loc[idOptStrat])
# ut.writeData(genlStratMinsData, "general_strat_mins_data")


compareParamData = ut.readData("compare_param_data", "dynamic_pred")
a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = compareParamData.loc['restored']
print(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
Aj, Bj, Aa, Ba = gs.genGenlStrats(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
genlStratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
ut.writeData(genlStratData, "general_strat_data_rstd")
p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)


# pqrsData = pd.DataFrame({'p': p,'q': q,'r': r,'s': s})
# _p, _q, _r, _s = pqrsData.loc[1043, ['p','q','r','s']]
# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=0.1)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=1000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# _FLim, err = gs.calcFLim(_p, _q, _r, _s, F0=-1000)
# print(_FLim)
# z1, z2 = gs.calcZLim(_p, _q, _r, _s, _FLim)
# print(z1, z2)
# gs.chkFLim(_p, _q, _r, _s, _FLim, z1, z2)
# print()

# pqrsRow = pqrsData.loc[[1043]]
# stratRow = genlStratData.loc[[1043]]
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=5000, tParts=100000, z0=0.001, F0=0.1)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)
# rawPopData = gs.calcPopDynamics(pqrsRow, tMax=5000, tParts=100000, z0=0.001, F0=1000)
# stratPopData, FLim = gs.analyzePopDynamics(stratRow, rawPopData, 0.01)


genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
print(genlStratMinsData.loc[idOptStrat])
ut.writeData(genlStratMinsData, "general_strat_mins_rstd_data")

_Aj, _Bj, _Aa, _Ba = genlStratData.loc[idOptStrat, 'Aj':'Ba']
gui.compareStratSins(-34.58, -3.29, -83.32, -51.57, _Aj, _Bj, _Aa, _Ba)
plt.show()

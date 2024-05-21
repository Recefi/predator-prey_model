import time
import gc
import pandas as pd

import libs.gen_selection as gs
import libs.utility as ut
import libs.graphical_interface as gui

Aj, Bj, Aa, Ba = gs.genGenlStrats()
genlStratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
ut.writeData(genlStratData, "general_strat_data")


# p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba)
# #pqrsData = pd.DataFrame({'p': p,'q': q,'r': r,'s': s})
# genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
# print(genlStratMinsData.loc[idOptStrat])
# ut.writeData(genlStratMinsData, "general_strat_mins_data")


compareParamData = ut.readData("compare_param_data", "dynamic_pred")
#compareParamData = ut.readData("compare_param_data.best", "dynamic_pred")
a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = compareParamData.loc['restored']
print(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)

p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
print(genlStratMinsData.loc[idOptStrat])
ut.writeData(genlStratMinsData, "general_strat_mins_rstd_data")
gui.stratSinsById(genlStratData, idOptStrat)

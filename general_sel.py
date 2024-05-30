import time
import gc
import pandas as pd
import matplotlib.pyplot as plt

import libs.gen_selection as gs
import libs.utility as ut
import libs.graphical_interface as gui
import find_param as fp

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
# strat = fp.stratByParam(a_j, a_a, b_j, b_a, g_j, g_a, d_j, d_a)
# print(strat.x, strat.fun)
#Aj, Bj, Aa, Ba = gs.genGenlStrats(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)
Aj, Bj, Aa, Ba = gs.genGenlStratsAll(a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a,
                                        Aj_left=-90, Aj_right=-30, Aj_step=2, Bj_step=1,
                                            Aa_left=-90, Aa_right=-30, Aa_step=4, Ba_step=4)

genlStratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
ut.writeData(genlStratData, "general_strat_data_rstd")
p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba, a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)

genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
print(genlStratMinsData.loc[idOptStrat])
ut.writeData(genlStratMinsData, "general_strat_mins_rstd_data")

_Aj, _Bj, _Aa, _Ba = genlStratData.loc[idOptStrat, 'Aj':'Ba']
gui.compareStratSins(-34.58, -3.29, -83.32, -51.57, _Aj, _Bj, _Aa, _Ba)
plt.show()

import time
import gc
import pandas as pd

import libs.gen_selection as gs
import libs.utility as ut

Aj, Bj, Aa, Ba = gs.genGenlStrats()
# start = time.time()
# genlStratData = pd.DataFrame({'Aj': Aj, 'Bj': Bj, 'Aa': Aa, 'Ba': Ba})
# ut.writeData(genlStratData, "general_strat_data")
# print ("write genl strats time: ", time.time() - start)
p, q, r, s = gs.calcGenlPqrsData(Aj, Bj, Aa, Ba)
#pqrsData = pd.DataFrame({'p': p,'q': q,'r': r,'s': s})
genlStratMinsData, idOptStrat = gs.genlFitMaxMin(Aj, Bj, Aa, Ba, p, q, r, s)
print(stratMinsData.loc[idOptStrat])

start = time.time()
ut.writeData(genlStratMinsData, "general_strat_mins_data")
print ("write genl mins time: ", time.time() - start)

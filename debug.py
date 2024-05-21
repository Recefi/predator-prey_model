import pandas as pd

import libs.utility as ut
import libs.gen_selection as gs

stratData = ut.readData("strat_data", "dynamic_pred")

mpData = ut.readData("mp_data", "dynamic_pred")
pqrsData = ut.readData("pqrs_data", "dynamic_pred")
stratFitData = gs.calcStratFitData(stratData, pqrsData, F=0.3414)
stratFitMpData = pd.concat([stratFitData, mpData.loc[:, 'M1':'M8']], axis="columns")
ut.writeData(stratFitMpData, "strat_fit_mp_data")

# stratFitMpData = ut.readData("strat_fit_mp_data")
# print(len(stratFitMpData.index))
# ut.writeData(stratData.loc[stratFitMpData.index], "strat_data", "dynamic_pred")

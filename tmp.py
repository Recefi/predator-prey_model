import matplotlib.pyplot as plt

import libs.graphical_interface as gui
import libs.gen_selection as gs
import libs.utility as ut


#gui.stratSins(-35, 3.93, -83, 49.32)
# gui.stratSins(-34.58, 3.29, -83.32, 51.57)
# plt.show()

stratData = gs.genStrats(10000, "beta")
# stratData = ut.readData("strat_data", "dynamic_pred")

stratData.loc[len(stratData.index)] = [-34.58, -3.29, -83.32, -51.57]

mpData, pqrsData = gs.calcMps(stratData)

stratFitData = gs.calcFitness(stratData, pqrsData, F=0.3383)
mpData = mpData.loc[stratFitData.index]
print("strats: ", len(stratFitData.index))

gui.mostOptStratSins_static(stratFitData, 3, 4)
plt.show()

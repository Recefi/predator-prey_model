import matplotlib.pyplot as plt

import libs.graphical_interface as gui
import libs.gen_selection as gs
import libs.utility as ut


#gui.stratSins(-35, 3.93, -83, 49.2)
#plt.show()


#stratData = gs.genStrats(100000, "beta")
stratData = ut.readData("strat_data", "dynamic_pred")

mpData, pqrsData = gs.calcMps(stratData)

stratFitData = gs.calcFitness(stratData, pqrsData, F=0.2878)
mpData = mpData.loc[stratFitData.index]
print("strats: ", len(stratFitData.index))

gui.mostOptStratSins_static(stratFitData, 3, 4)
plt.show()

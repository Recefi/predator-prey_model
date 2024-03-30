import libs.graphical_interface as gui
import libs.gen_selection as gs
import matplotlib.pyplot as plt

gui.stratSins(-35, 3.93, -83, 49.2)
#plt.show()


stratData = gs.genStrats(100000, "beta", by4=True)

mpData, pqrsData = gs.calcMps(stratData)

stratFitData = gs.calcFitness(stratData, pqrsData)
mpData = mpData.loc[stratFitData.index]
print("strats: ", len(stratFitData.index))

gui.mostOptStratSins(stratFitData, 3, 4)
plt.show()

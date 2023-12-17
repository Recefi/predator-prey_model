import argparse
import matplotlib.pyplot as plt
import time

import source.graphical_interface as gui
import source.utility as ut


parser = argparse.ArgumentParser()
parser.add_argument("callerName", type=str)
parser.add_argument("--show", default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

norm_selData = ut.readData("norm_sel_data", args.callerName)
norm_coefData = ut.readData("norm_coef_data", args.callerName)
norm_mlLams = norm_coefData['ml'].values

fig = gui.clfPlanes(norm_selData, norm_mlLams)

start = time.time()
if (args.show):
    plt.show()
    end = time.time()
    print ("show 8x8 planes time: ", end - start)
# !!! If at once, then doubled time !!!
else:
    ut.writeImage(fig, "8x8clfPlanes.png", args.callerName)
    plt.close()
    end = time.time()
    print ("write 8x8 planes time: ", end - start)

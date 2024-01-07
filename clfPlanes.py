import argparse
import matplotlib.pyplot as plt
import time

import libs.graphical_interface as gui
import libs.utility as ut


parser = argparse.ArgumentParser()
parser.add_argument("callerName", type=str)
parser.add_argument("--lam0", default=0.0, type=float)  # --lam0=...
parser.add_argument("--show", default=True, action=argparse.BooleanOptionalAction)  # --show | --no-show
args = parser.parse_args()

norm_selData = ut.readData("sel_data", args.callerName)
mpMaxs = ut.readData("mp_maxs_data", args.callerName).values[0]
norm_selData.loc[:,'M1':'M8M8'] = norm_selData.loc[:,'M1':'M8M8'] / mpMaxs
norm_coefData = ut.readData("coef_data", args.callerName)
norm_mlLams = [args.lam0] + norm_coefData.loc[-2].to_list()

fig = gui.clfPlanes(norm_selData[0:], norm_mlLams)

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

import sys
import os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
import time

import libs.graphical_interface as gui
import libs.gen_selection as gs
import libs.utility as ut


class resODE():
    def __init__(self, time, strats, err):
        self.time = time
        self.strats = strats
        self.err = err

def test1():
    resODE1 = []
    resODE2 = []
    resODE3 = []
    for i in range(1000):
        print("#"+str(i))
        print("-----------------------------------------------------------")
        stratData = gs.genStrats(100, "beta", ab=5)
        stratData.loc[len(stratData.index) - 1] = [-34.58, -3.29, -83.32, -51.57]
        mpData = gs.calcMpData(stratData)
        pqrsData = gs.calcPqrsData(mpData)

        start = time.time()
        rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=5001, z0=0.001, F0=0.001, _method='Radau')
        _time = time.time() - start
        print ("(Radau) calc pop dynamics: ", _time)
        start = time.time()
        stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
        print ("analyze pop dynamics: ", time.time() - start)
        strats = len(stratPopData.index)
        print("strats: ", strats)
        stratPopFitData = gs.calcStratFitData(stratPopData, pqrsData.loc[stratPopData.index], F=0.3383)
        err = gs.checkRanking(stratPopFitData)
        print("ranking error: ", err)
        resODE1.append(resODE(_time, strats, err))
        print("-----------------------------")
        start = time.time()
        rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=5001, z0=0.001, F0=0.001, _method='BDF')
        _time = time.time() - start
        print ("(BDF) calc pop dynamics: ", _time)
        start = time.time()
        stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
        print ("analyze pop dynamics: ", time.time() - start)
        strats = len(stratPopData.index)
        print("strats: ", strats)
        stratPopFitData = gs.calcStratFitData(stratPopData, pqrsData.loc[stratPopData.index], F=0.3383)
        err = gs.checkRanking(stratPopFitData)
        print("ranking error: ", err)
        resODE2.append(resODE(_time, strats, err))
        print("-----------------------------")
        start = time.time()
        rawPopData = gs.calcPopDynamics(pqrsData, tMax=5000, tParts=5001, z0=0.001, F0=0.001, _method='LSODA')
        _time = time.time() - start
        print ("(LSODA) calc pop dynamics: ", _time)
        start = time.time()
        stratPopData, FLim = gs.analyzePopDynamics(stratData, rawPopData, 0.01)
        print ("analyze pop dynamics: ", time.time() - start)
        strats = len(stratPopData.index)
        print("strats: ", strats)
        stratPopFitData = gs.calcStratFitData(stratPopData, pqrsData.loc[stratPopData.index], F=0.3383)
        err = gs.checkRanking(stratPopFitData)
        print("ranking error: ", err)
        resODE3.append(resODE(_time, strats, err))
        print("-----------------------------------------------------------")
    print("Radau: ", np.mean([r.time for r in resODE1]), np.mean([r.strats for r in resODE1]),
                                                                                    np.mean([r.err for r in resODE1]))
    print("BDF: ", np.mean([r.time for r in resODE2]), np.mean([r.strats for r in resODE2]),
                                                                                    np.mean([r.err for r in resODE2]))
    print("LSODA: ", np.mean([r.time for r in resODE3]), np.mean([r.strats for r in resODE3]),
                                                                                    np.mean([r.err for r in resODE3]))
    print("-----------------------------------------------------------")

test1()

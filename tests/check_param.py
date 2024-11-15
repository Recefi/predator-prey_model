import sys
import os
sys.path.insert(0, os.path.abspath(".."))

import libs.utility as ut

#rngLst = list(range(1, 10))
#rngLst.remove(6)
rngLst = list(range(1, 4))

dataIdxsList = []
for i in rngLst:
    checkParamData = ut.readData("check_param_data", callerName="dynamic_pred", subDirsName="check_param/"+str(i))
    checkParamData = checkParamData[checkParamData['odeRes'] == 99]
    dataIdxsList.append(checkParamData.index.to_numpy())
    print(dataIdxsList[-1])

init = 0
for i in range(1, len(dataIdxsList)):
    if (dataIdxsList[i].size > dataIdxsList[init].size):
        init = i

res = []
rngLst = list(range(len(dataIdxsList)))
rngLst.remove(init)
for initIdx in dataIdxsList[init]:
    isOver = False
    for i in rngLst:
        if isOver:
            break
        for idx in dataIdxsList[i]:
            if (idx > initIdx):
                isOver = True
                break
            if (idx == initIdx):
                break
    if not isOver:
        res.append(initIdx)

print(res)

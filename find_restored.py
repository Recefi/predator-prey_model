import libs.utility as ut

for i in range(1000):
    with open("dynamic_pred.py") as iter:
        exec(iter.read())
    _compareParamData = ut.readData("_compare_param_data", "<string>")
    params = _compareParamData.iloc[-1]
    #print(params)
    if all(params > 0):
        break

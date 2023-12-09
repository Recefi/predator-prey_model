import pandas as pd


def readData(fileName):
    data = pd.read_csv("csv/" + fileName + ".csv", index_col=0)
    return data

def writeData(data, fileName):
    data.to_csv("csv/" + fileName + ".csv", index=True)
    
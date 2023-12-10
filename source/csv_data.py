import pandas as pd
import inspect
import os


def getCallerName():
    callerFrame = inspect.stack()[2]
    callerFullFilename = callerFrame.filename
    callerName = os.path.splitext(os.path.basename(callerFullFilename))[0]
    return callerName

def checkDirs(callerName):
    if not os.path.exists("csv"):
        os.mkdir("csv")
    if not os.path.exists("csv/" + callerName):
        os.mkdir("csv/" + callerName)

def readData(fileName):
    callerFrame = inspect.stack()[1]
    callerFullFilename = callerFrame.filename
    callerFilename = os.path.basename(callerFullFilename)  # get rid of the directory
    callerName = os.path.splitext(callerFilename)[0]  # split filename and extension

    checkDirs(callerName)
    data = pd.read_csv("csv/" + callerName + "/" + fileName + ".csv", index_col=0)
    return data

def writeData(data, fileName):
    callerName = getCallerName()
    
    checkDirs(callerName)
    data.to_csv("csv/" + callerName + "/" + fileName + ".csv", index=True)
    
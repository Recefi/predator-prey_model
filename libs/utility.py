import pandas as pd
import inspect
import os


def getCallerName(lvl = 2):
    callerFrame = inspect.stack()[lvl]
    callerFullFilename = callerFrame.filename
    callerName = os.path.splitext(os.path.basename(callerFullFilename))[0]
    return callerName

def chkDir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)

def readData(fileName, callerName="", subDirsName=""):
    if not callerName:
        callerFrame = inspect.stack()[1]
        callerFullFilename = callerFrame.filename
        callerFilename = os.path.basename(callerFullFilename)  # get rid of the directory
        callerName = os.path.splitext(callerFilename)[0]  # split filename and extension
    if subDirsName:
        callerName += ("/" + subDirsName)

    chkDir("csv/" + callerName)
    data = pd.read_csv("csv/" + callerName + "/" + fileName + ".csv", index_col=0)
    return data

def writeData(data, fileName, callerName="", subDirsName=""):
    if not callerName: callerName = getCallerName()
    if subDirsName: callerName += ("/" + subDirsName)

    chkDir("csv/" + callerName)
    data.to_csv("csv/" + callerName + "/" + fileName + ".csv", index=True)

def writeImage(fig, fileName, callerName="", subDirsName=""):
    if not callerName: callerName = getCallerName()
    if subDirsName: callerName += ("/" + subDirsName)

    chkDir("images", callerName)
    fig.savefig("images/" + callerName + "/" + fileName)

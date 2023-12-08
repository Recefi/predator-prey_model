import pandas as pd


def readData(fileName):
    data = pd.read_csv("csv/" + fileName + ".csv", index_col=0)
    return data

def writeData(data, fileName):
    data.to_csv("csv/" + fileName + ".csv", index=True)

def parseStratData(stratData):
    """
    Возвращает: A_j, B_j, A_a, B_a 
        в виде pandas series (с сохранением исходных индексов и не только)
    """
    A_j = stratData['Aj']
    B_j = stratData['Bj']
    A_a = stratData['Aa']
    B_a = stratData['Ba']
    return A_j, B_j, A_a, B_a

def collectStratData(A_j, B_j, A_a, B_a):
    """Собирает данные стратегий"""
    stratData = pd.DataFrame({'Aj': A_j, 'Bj': B_j, 'Aa': A_a, 'Ba': B_a})
    return stratData

def parseMpData(mpData):
    """
    Возвращает: Mps, OrigIndxs
        OrigIndxs[индекс Mps] = исходный индекс
    """
    Mps = mpData.values
    OrigIndxs = mpData.index
    return Mps, OrigIndxs

def collectMpData(Mps, OrigIndxs):
    """Собирает данные макропараметров"""
    cols = []
    for i in range(1, 9):
        cols.append('M'+str(i))
    for i in range(1, 9):
        for j in range(i, 9):
            cols.append('M'+str(i) + 'M'+str(j))

    mpData = pd.DataFrame(Mps, columns=cols, index=OrigIndxs)
    return mpData
    
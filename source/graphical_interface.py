import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import source.csv_data as cd


def showSin(Aj, Bj, Aa, Ba):
    fig, ax = plt.subplots()

    xj = np.linspace(0, 1)
    yj = Aj + Bj * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="blue", label="Молодые особи")

    xa = np.linspace(0, 1)
    ya = Aa + Ba * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="red", label="Взрослые особи")

    ax.legend()
    plt.show()

def showOptSin(stratFitData):
    maxFitId = stratFitData['fit'].idxmax()
    Aj = stratFitData['Aj'].loc[maxFitId]
    Bj = stratFitData['Bj'].loc[maxFitId]
    Aa = stratFitData['Aa'].loc[maxFitId]
    Ba = stratFitData['Ba'].loc[maxFitId]
    showSin(Aj, Bj, Aa, Ba)

def showAllSins(stratData):
    Aj = stratData['Aj']
    Bj = stratData['Bj']
    Aa = stratData['Aa']
    Ba = stratData['Ba']

    fig, ax = plt.subplots()
    for i in Aj.index:
        xj = np.linspace(0, 1)
        yj = Aj[i] + Bj[i] * np.cos(2 * np.pi * xj)
        ax.plot(xj, yj, c="blue")

        xa = np.linspace(0, 1)
        ya = Aa[i] + Ba[i] * np.cos(2 * np.pi * xa)
        ax.plot(xa, ya, c="red")

    plt.show()

def showComparisonSins(stratData, maxTrueFitId, maxRestrFitId):
    fig, ax = plt.subplots()
    trueOptStrat = stratData.loc[maxTrueFitId]
    restrOptStrat = stratData.loc[maxRestrFitId]

    xj = np.linspace(0, 1)
    yj = trueOptStrat['Aj'] + trueOptStrat['Bj'] * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="blue", label="Молодые (по исх. функции)")
    xa = np.linspace(0, 1)
    ya = trueOptStrat['Aa'] + trueOptStrat['Ba'] * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="red", label="Взрослые (по исх. функции)")

    yj = restrOptStrat['Aj'] + restrOptStrat['Bj'] * np.cos(2 * np.pi * xj)
    ax.plot(xj, yj, c="green", label="Молодые (по восст. функции)")
    ya = restrOptStrat['Aa'] + restrOptStrat['Ba'] * np.cos(2 * np.pi * xa)
    ax.plot(xa, ya, c="orange", label="Взрослые (по восст. функции)")

    ax.legend()
    plt.show()

def showPopDynamics(rawData):
    n = int(len(rawData.index)/2)

    j_data = rawData.iloc[:n]
    a_data = rawData.iloc[n:2*n]
    F_data = rawData.loc['F']

    fig1, ax1 = plt.subplots()
    j_data.T.plot(ax=ax1, title="Молодые особи", xlabel="t", legend=False)
    aj_yMax = j_data.max().max()

    fig2, ax2 = plt.subplots()
    a_data.T.plot(ax=ax2, title="Взрослые особи", xlabel="t", legend=False)
    a_yMax = a_data.max().max()
    if (a_yMax > aj_yMax):
        aj_yMax = a_yMax
    
    fig3, ax3 = plt.subplots()
    F_data.T.plot(ax=ax3, title="Хищник", xlabel="t", legend=False)

    ax1.set_ylim([0, aj_yMax*1.1])
    ax2.set_ylim([0, aj_yMax*1.1])
    ax3.set_ylim([0, F_data.max()*1.1])
    plt.show()

def showHistMps(mpData):
    mpData.loc[:,'M1':'M8'].hist(layout=(2, 4), figsize=(12, 6))
    plt.tight_layout()
    plt.show()

def showCorrMps(mpData):
    corrMatr=np.round(np.corrcoef(mpData.loc[:,'M1':'M8'].T.values),2)
    
    fig, ax = plt.subplots()
    im = ax.imshow(corrMatr)

    ax.set_xticks(np.arange(8), labels=mpData[['M1','M2','M3','M4','M5','M6','M7','M8']].columns)
    ax.set_yticks(np.arange(8), labels=mpData.loc[:,'M1':'M8'].columns)

    for i in range(8):
        for j in range(8):
            ax.text(j, i, corrMatr[i, j], ha="center", va="center", color="r")

    fig.tight_layout()
    plt.show()

def drawClfOneSlice(selData, mlLams, intercept, i, j):
    """(i,j)<->(y,x)"""
    x1 = selData['M'+str(j+1)].values
    x2 = selData['M'+str(i+1)].values
    y = selData['class'].values
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel('M'+str(j+1))
    ax.set_ylabel('M'+str(i+1))
    s = ax.scatter(x1, x2, c=y, s=5, cmap=plt.cm.Paired, alpha=0.5)

    x_visual = np.linspace(-1,1)
    y_visual = -(mlLams[j] / mlLams[i]) * x_visual - intercept / mlLams[i]
    ax.plot(x_visual, y_visual, color="blue", label="ML")
    
    leg1 = ax.legend(handles=s.legend_elements(alpha=1)[0], labels=s.legend_elements(alpha=1)[1], loc="lower left", title="Class", draggable=True)
    ax.add_artist(leg1)  # for ax.plot legend
    ax.legend(loc="upper right", title="Hyperplane", draggable=True)  # for ax.plot legend

    fig.tight_layout()
    plt.draw()

def showClfSlices(selData, mlLams, intercept):
    X = selData.loc[:,'M1':'M8'].values
    y = selData['class'].values

    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):  # (i,j)<->(y,x), тогда: по строкам - i, по столбцам - j
            ax[i][j].set(xlim=(-1, 1), ylim=(-1, 1))
            if j==0:
                ax[i][j].set_ylabel('M'+str(i+1))
            if i==7:
                ax[i][j].set_xlabel('M'+str(j+1))
            if i<7:
                ax[i][j].set_xticks([])
                ax[i][j].set_xticks([], minor=True)
            if j>0:
                ax[i][j].set_yticks([])
                ax[i][j].set_yticks([], minor=True)
            if i!=j:
                ax[i][j].scatter(X[:, j], X[:, i], c=y, s=1, cmap=plt.cm.Paired, alpha=0.5)

                x_visual = np.linspace(-1, 1)
                y_visual = -(mlLams[j] / mlLams[i]) * x_visual - intercept / mlLams[i]
                ax[i][j].plot(x_visual, y_visual, color="blue")

                # уравнение гиперплоскости:
                # intercept + lam[0]*M1 + lam[1]*M2 + lam[2]*M3 + ... + lam[43]*M8M8 = 0  ||  lam[0]*M1 + lam[1]*M2 + lam[2]*M3 + ... + lam[43]*M8M8 = b

                #   пусть:
                # W := (w1,w2), X := (x,y), b := -w0
                #   тогда:
                # w0 + <W,X> = 0 ---> W^T * X - b = 0 ---> (w1,w2)*(x,y)^T - b = 0 ---> w1*x + w2*y - b = 0 ---> y = -(w1/w2)*x + b/w2 ---> y = -(w1/w2) - w0/w2
                #   в данном случае:
                # w := (lam[1],lam[2],...,lam[43])^T, x := (M1,M2,...,M8M8), b := -intercept
                #   тогда для двухмерной проекции:
                # w^T * x - b = 0 ---> (lam[0],lam[1]) * (M1, M2)^T - b = 0 ---> lam[0]*M1 + lam[1]*M2 + intercept = 0 ---> M2 = -(lam[0]/lam[1])*M1 - intercept/lam[1]
                #   для трехмерной проекции итд:
                # lam[0]*M1 + lam[1]*M2 + lam[2]*M3 + ... + intercept = 0 ---> ...

    fig.tight_layout()
    plt.show()


def drawRegLine(x, y):
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.legend()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.draw()
    return intercept, slope, xlim, ylim

def cleanRegLine(fitData, xName, yName, a, b, shift):
    x0 = fitData[xName]
    y0 = fitData[yName]
    indexes = []
    for i in x0.index:
        y1 = a - shift + b*x0[i]
        y2 = a + shift + b*x0[i]
        if (y0[i] > y1 and y0[i] < y2):
            indexes.append(i)
    return fitData.drop(indexes)

def drawLimRegLine(x, y, xlim, ylim):
    slope, intercept, r, p, stderr = stats.linregress(x, y)

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.scatter(x, y, s=3, label = str(len(x))+" points", color="red")
    ax.plot(x, intercept + slope * x, label = f'corr={r:.2f}', color="blue")
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.legend()

    plt.draw()

def fixCorr(fitData, xName, yName, shift):
    """
    исправление корреляции между xName и yName 
        удалением стратегий с отступами от линии регрессии на shift
    """
    a, b, xlim, ylim = draw_regLine(fitData[xName], fitData[yName])
    fitData = clean_regLine(fitData, xName, yName, a, b, shift)
    draw_limRegLine(fitData[xName], fitData[yName], xlim, ylim)

    return fitData


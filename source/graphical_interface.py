import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time

import source.utility as ut
import source.param as param


def showSin(Aj, Bj, Aa, Ba):
    fig, ax = plt.subplots()

    x = np.linspace(0, 1)
    yj = Aj + Bj * np.cos(2 * np.pi * x)
    ax.plot(x, yj, c="blue", label="Молодые особи")
    ya = Aa + Ba * np.cos(2 * np.pi * x)
    ax.plot(x, ya, c="red", label="Взрослые особи")

    ax.legend()
    plt.show()

def showOptSin(stratFitData):
    maxFitId = stratFitData['fit'].idxmax()
    Aj = stratFitData['Aj'].loc[maxFitId]
    Bj = stratFitData['Bj'].loc[maxFitId]
    Aa = stratFitData['Aa'].loc[maxFitId]
    Ba = stratFitData['Ba'].loc[maxFitId]
    showSin(Aj, Bj, Aa, Ba)

def showMostOptSins(stratFitData, rows, cols):
    optStratData = stratFitData.sort_values(by=['fit'], ascending=False).head(12)
    indxs = optStratData.index
    fit = optStratData['fit'].values
    Aj = optStratData['Aj'].values
    Bj = optStratData['Bj'].values
    Aa = optStratData['Aa'].values
    Ba = optStratData['Ba'].values

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    x = np.linspace(0, 1)
    for i in range(rows):
        for j in range(cols):
            yj = Aj[i*4+j] + Bj[i*4+j] * np.cos(2 * np.pi * x)
            ax[i][j].plot(x, yj, c="blue", label="Aj: "+str(np.round(Aj[i*4+j], 2))+"\nBj: "+str(np.round(Bj[i*4+j], 2)))
            # ax[i][j].plot(x, yj, c="blue")
            ya = Aa[i*4+j] + Ba[i*4+j] * np.cos(2 * np.pi * x)
            ax[i][j].plot(x, ya, c="red", label="Aa: "+str(np.round(Aa[i*4+j], 2))+"\nBa: "+str(np.round(Ba[i*4+j], 2)))
            # ax[i][j].plot(x, ya, c="red")
            ax[i][j].set_title("strat: " + str(indxs[i*4+j]) + "\n" + "fit: " + str(fit[i*4+j]))
            ax[i][j].set_ylim(-param.D - 0.5, 0.5)
            ax[i][j].legend()

    fig.tight_layout()
    plt.show()

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

    x = np.linspace(0, 1)
    yj = trueOptStrat['Aj'] + trueOptStrat['Bj'] * np.cos(2 * np.pi * x)
    ax.plot(x, yj, c="blue", label="Молодые (по исх. функции)")
    ya = trueOptStrat['Aa'] + trueOptStrat['Ba'] * np.cos(2 * np.pi * x)
    ax.plot(x, ya, c="red", label="Взрослые (по исх. функции)")

    yj = restrOptStrat['Aj'] + restrOptStrat['Bj'] * np.cos(2 * np.pi * x)
    ax.plot(x, yj, c="green", label="Молодые (по восст. функции)")
    ya = restrOptStrat['Aa'] + restrOptStrat['Ba'] * np.cos(2 * np.pi * x)
    ax.plot(x, ya, c="orange", label="Взрослые (по восст. функции)")

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

def drawClf2dPlane(selData, lams, lam0, i, j):
    """(i,j)<->(y,x)"""
    x1 = selData['M'+str(j+1)].values
    x2 = selData['M'+str(i+1)].values
    y = selData['class'].values
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel('M'+str(j+1))
    ax.set_ylabel('M'+str(i+1))
    s = ax.scatter(x1, x2, c=y, s=5, cmap=plt.cm.Paired, alpha=0.5)

    x_visual = np.linspace(-1,1)
    y_visual = -(lams[j] / lams[i]) * x_visual - lam0 / lams[i]
    ax.plot(x_visual, y_visual, color="blue", label="ML")
    
    leg1 = ax.legend(*s.legend_elements(alpha=1), loc="lower left", title="Class", draggable=True)
    ax.add_artist(leg1)  # for ax.plot legend
    ax.legend(loc="upper right", title="Hyperplane", draggable=True)  # for ax.plot legend

    fig.tight_layout()
    plt.draw()

def showClfPlanes(selData, lams, lam0):
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
                y_visual = -(lams[j] / lams[i]) * x_visual - lam0 / lams[i]
                ax[i][j].plot(x_visual, y_visual, color="blue")

                # в отличии от исх.свертки макропараметров уравнение восст.гиперплоскости скорее всего содержит lam0!=0, если точность класс-ра не 100%:
                    # lam0 + lams[0]*M1 + lams[1]*M2 + lams[2]*M3 + ... + lams[43]*M8M8 = 0  ||  lams[0]*M1 + lams[1]*M2 + lams[2]*M3 + ... + lams[43]*M8M8 = b
                # lam0 следует использовать при демонстрации рез-та обучения, но не в дальнейшем!

                # пусть:
                    # W := (w1,w2), X := (x,y), b := -w0
                # тогда:
                    # w0 + <W,X> = 0 ---> W^T * X - b = 0 ---> (w1,w2)*(x,y)^T - b = 0 ---> w1*x + w2*y - b = 0 ---> y = -(w1/w2)*x + b/w2 ---> y = -(w1/w2) - w0/w2
                # в данном случае:
                    # w := (lams[1],lams[2],...,lams[43])^T, x := (M1,M2,...,M8M8), b := -lam0
                # тогда для двухмерной проекции:
                    # w^T * x - b = 0 ---> (lams[0],lams[1]) * (M1, M2)^T - b = 0 ---> lams[0]*M1 + lams[1]*M2 + lam0 = 0 ---> M2 = -(lams[0]/lams[1])*M1 - lam0/lams[1]
                # для трехмерной проекции итд:
                    # lams[0]*M1 + lams[1]*M2 + lams[2]*M3 + ... + lam0 = 0 ---> ...
    
    fig.tight_layout()

    # start = time.time()
    # ut.writeImage(fig, "8x8clfPlanes.png")
    # end = time.time()
    # print ("write 8x8 planes time: ", end - start)
    # plt.close()
    # !!! If at once, then the time doubles !!!
    start = time.time()
    plt.show()
    end = time.time()
    print ("show 8x8 planes time: ", end - start)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import time

import source.utility as ut
import source.param as param


def stratSins(Aj, Bj, Aa, Ba):
    fig, ax = plt.subplots()

    x = np.linspace(0, 1)
    yj = Aj + Bj * np.cos(2 * np.pi * x)
    ax.plot(x, yj, c="blue", label="Молодые особи")
    ya = Aa + Ba * np.cos(2 * np.pi * x)
    ax.plot(x, ya, c="red", label="Взрослые особи")

    ax.legend()

def optStratSins(stratFitData):
    maxFitId = stratFitData['fit'].idxmax()
    Aj = stratFitData['Aj'].loc[maxFitId]
    Bj = stratFitData['Bj'].loc[maxFitId]
    Aa = stratFitData['Aa'].loc[maxFitId]
    Ba = stratFitData['Ba'].loc[maxFitId]
    stratSin(Aj, Bj, Aa, Ba)

def mostOptStratSins(stratFitData, rows, cols):
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

def allSins(stratData):
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

def comparisonSins(stratData, maxTrueFitId, maxRestrFitId):
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

def popDynamics(rawData):
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

def histMps(mpData):
    mpData.loc[:,'M1':'M8'].hist(layout=(2, 4), figsize=(12, 6), bins=np.linspace(-1,1,100))
    plt.tight_layout()

def corrMps(mpData):
    corrMatr=np.round(np.corrcoef(mpData.loc[:,'M1':'M8'].T.values),2)
    
    fig, ax = plt.subplots()
    im = ax.imshow(corrMatr)

    ax.set_xticks(np.arange(8), labels=mpData[['M1','M2','M3','M4','M5','M6','M7','M8']].columns)
    ax.set_yticks(np.arange(8), labels=mpData.loc[:,'M1':'M8'].columns)

    for i in range(8):
        for j in range(8):
            ax.text(j, i, corrMatr[i, j], ha="center", va="center", color="r")

    fig.tight_layout()

def clf2dPlane(selData, lams, M1, M2):
    x_x = selData[M1].values
    x_y = selData[M2].values
    y = selData['class'].values
    lam0 = lams[0]
    lam1 = lams[selData.columns == M1]
    lam2 = lams[selData.columns == M2]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel(M1)
    ax.set_ylabel(M2)
    ax.set(xlim=(-1, 1), ylim=(-1, 1))
    s = ax.scatter(x_x, x_y, c=y, cmap=ListedColormap(["xkcd:tomato", "xkcd:lightblue"]), s=5, alpha=0.1)

    x_visual = np.linspace(-1,1)
    y_visual = -(lam1/lam2)*x_visual - lam0/lam2
    ax.plot(x_visual, y_visual, color="blue", label="ML")
    
    leg1 = ax.legend(*s.legend_elements(alpha=1), loc="lower left", title="Class", draggable=True)
    ax.add_artist(leg1)  # needs for ax.plot legend
    ax.legend(loc="upper right", title="Hyperplane", draggable=True)  # needs for ax.plot legend

    fig.tight_layout()

def clf3dPlaneMPL(selData, lams, M1, M2, M3, elevation=30, azimuth=-60):
    x_x = selData[M1].values
    x_y = selData[M2].values
    x_z = selData[M3].values
    y = selData['class'].values
    lam0 = lams[0]
    lam1 = lams[selData.columns == M1]
    lam2 = lams[selData.columns == M2]
    lam3 = lams[selData.columns == M3]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(M1)
    ax.set_ylabel(M2)
    ax.set_zlabel(M3)
    s = ax.scatter(x_x, x_y, x_z, c=y, cmap=ListedColormap(["xkcd:tomato", "xkcd:lightblue"]), alpha=0.1)

    tmp = np.linspace(-1,1)
    x_visual, y_visual = np.meshgrid(tmp,tmp)
    z_visual = lambda x_vis,y_vis: -(lam1/lam3)*x_vis - (lam2/lam3)*y_vis - lam0/lam3
    ax.plot_surface(x_visual, y_visual, z_visual(x_visual, y_visual))

    ax.view_init(elevation, azimuth)

    ax.legend(*s.legend_elements(alpha=1), title="Class")
    fig.tight_layout()

def clfPlanes(selData, lams):
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
                ax[i][j].scatter(X[:, j], X[:, i], c=y, cmap=ListedColormap(["xkcd:tomato", "xkcd:lightblue"]), s=5, alpha=0.1)

                x_visual = np.linspace(-1, 1)
                y_visual = -(lams[j+1]/lams[i+1])*x_visual - lams[0]/lams[i+1]
                ax[i][j].plot(x_visual, y_visual, color="blue")

                # в отличии от исх.свертки макропар-ов ур-е восст.гиперплоскости скорее всего содержит lam0!=0, если точность класс-ра не 100% и ему не сообщили считать lam0=0:
                    # lam0 + lam1*M1 + lam2*M2 + lam3*M3 + ... + lam44*M8M8 = 0  ||  lam1*M1 + lam2*M2 + lam3*M3 + ... + lam44*M8M8 = b
                # lam0 следует использовать при демонстрации рез-та обучения, но не в дальнейшем!

                # в данном случае:
                    # w := (lam1,lam2,...,lam44)^T, x := (M1,M2,...,M8M8), b := -lam0
                # тогда для двухмерной проекции:
                    # <w,x> - b = 0 ---> w^T * x - b = 0 ---> (lam1,lam2) * (M1, M2)^T - b = 0 ---> lam1*M1 + lam2*M2 + lam0 = 0 ---> M2 = -(lam1/lam2)*M1 - lam0/lam2
                # для трехмерной проекции итд:
                    # lam1*M1 + lam2*M2 + lam3*M3 + ... + lam0 = 0 ---> ...
    
    fig.tight_layout()
    return fig

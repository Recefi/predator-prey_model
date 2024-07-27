import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import time

import libs.utility as ut
import libs.param as param


from matplotlib.colors import ListedColormap
cMap = ListedColormap(["xkcd:tomato", "deepskyblue"])

def stratSins(Aj, Bj, Aa, Ba, ax=None, c_j="blue", c_a="red", l_j="Молодые особи", l_a="Взрослые особи"):
    if ax is None:
        fig, ax = plt.subplots()

    x = np.linspace(0, 1)
    yj = Aj + Bj * np.cos(2 * np.pi * x)
    ax.plot(x, yj, c=c_j, label=l_j)
    ya = Aa + Ba * np.cos(2 * np.pi * x)
    ax.plot(x, ya, c=c_a, label=l_a)

    ax.legend()

def pointsCalanus_1(ax=None, c_j="blue", c_a="red", m='o', l_j="", l_a=""):
    if ax is None:
        fig, ax = plt.subplots()

    t = [[0.06, 0.19, 0.27, 0.38, 0.55, 0.7, 0.82, 0.98], [0.06, 0.19, 0.27, 0.38, 0.55, 0.7, 0.82, 0.98]]
    d = [[-33.5, -39, -33, -31, -41, -40, -31, -34], [-45, -32.75, -111, -125, -128, -105, -39.5, -32.5]]

    ax.scatter(t[0], d[0], color=c_j, marker=m, label=l_j)
    ax.scatter(t[1], d[1], color=c_a, marker=m, label=l_a)
    return ax

def pointsCalanus_2(ax=None, c_j="blue", c_a="red", m='o', l_j="", l_a=""):
    if ax is None:
        fig, ax = plt.subplots()

    t = [[0.56, 0.69, 0.77, 0.88, 0.05, 0.2, 0.32, 0.48], [0.56, 0.69, 0.77, 0.88, 0.05, 0.2, 0.32, 0.48]]
    d = [[-33.5, -39, -33, -31, -41, -40, -31, -34], [-45, -32.75, -111, -125, -128, -105, -39.5, -32.5]]

    ax.scatter(t[0], d[0], color=c_j, marker=m, label=l_j)
    ax.scatter(t[1], d[1], color=c_a, marker=m, label=l_a)
    return ax

def pointsPseudocalanus(ax=None, c_j="blue", c_a="red", m='o', l_j="", l_a=""):
    if ax is None:
        fig, ax = plt.subplots()

    t = [[0.06, 0.19, 0.27, 0.38, 0.55, 0.7, 0.82, 0.98], [0.06, 0.19, 0.27, 0.38, 0.55, 0.7, 0.82, 0.98]]
    d = [[-29, -37, -32.5, -36.5, -35, -34, -40, -33.5], [-33, -40, -105, -113, -119, -91, -60, -34.5]]

    ax.scatter(t[0], d[0], color=c_j, marker=m, label=l_j)
    ax.scatter(t[1], d[1], color=c_a, marker=m, label=l_a)
    return ax

def compareStratSins(Aj_1, Bj_1, Aa_1, Ba_1, Aj_2, Bj_2, Aa_2, Ba_2, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    x = np.linspace(0, 1)
    y = Aj_1 + Bj_1 * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="blue", label="Молодые особи")
    y = Aa_1 + Ba_1 * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="red", label="Взрослые особи")
    y = Aj_2 + Bj_2 * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="green", label="Молодые особи (восст.)")
    y = Aa_2 + Ba_2 * np.cos(2 * np.pi * x)
    ax.plot(x, y, c="orange", label="Взрослые особи (восст.)")

    ax.legend()

def stratSinsById(stratData, id, ax=None):
    Aj = stratData.loc[id, 'Aj']
    Bj = stratData.loc[id, 'Bj']
    Aa = stratData.loc[id, 'Aa']
    Ba = stratData.loc[id, 'Ba']
    stratSins(Aj, Bj, Aa, Ba, ax)

def compareStratSinsById(stratData, id_1, id_2, ax=None):
    Aj_1, Bj_1, Aa_1, Ba_1 = stratData.loc[id_1, 'Aj':'Ba']
    Aj_2, Bj_2, Aa_2, Ba_2 = stratData.loc[id_2, 'Aj':'Ba']
    compareStratSins(Aj_1, Bj_1, Aa_1, Ba_1, Aj_2, Bj_2, Aa_2, Ba_2, ax)

def optStratSins(stratFitData, key='fit'):
    optPntId = stratFitData[key].idxmax()
    stratSinsById(stratFitData, optPntId)

def mostOptStratSins(stratFitData, rows, cols, key='fit', title=""):
    optStratData = stratFitData.sort_values(by=[key], ascending=False).head(rows*cols)
    indxs = optStratData.index
    fit = optStratData[key].values
    Aj = optStratData['Aj'].values
    Bj = optStratData['Bj'].values
    Aa = optStratData['Aa'].values
    Ba = optStratData['Ba'].values

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    x = np.linspace(0, 1)
    for i in range(rows):
        for j in range(cols):
            k = i*cols+j
            yj = Aj[k] + Bj[k] * np.cos(2 * np.pi * x)
            ax[i][j].plot(x,yj, c="blue", label="Aj: "+str(np.round(Aj[k], 2))+"\nBj: "+str(np.round(Bj[k], 2)))
            # ax[i][j].plot(x, yj, c="blue")
            ya = Aa[k] + Ba[k] * np.cos(2 * np.pi * x)
            ax[i][j].plot(x,ya, c="red", label="Aa: "+str(np.round(Aa[k], 2))+"\nBa: "+str(np.round(Ba[k], 2)))
            # ax[i][j].plot(x, ya, c="red")
            ax[i][j].set_title("strat: " + str(indxs[k]) + "\n" + key+": " + str(fit[k]))
            ax[i][j].set_ylim(-param.D - 0.5, 0.5)
            ax[i][j].legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()

def allStratSins(stratData):
    Aj = stratData['Aj']
    Bj = stratData['Bj']
    Aa = stratData['Aa']
    Ba = stratData['Ba']

    fig, ax = plt.subplots()
    for i in Aj.index:
        x = np.linspace(0, 1)
        yj = Aj[i] + Bj[i] * np.cos(2 * np.pi * x)
        ax.plot(x, yj, c="blue")
        ya = Aa[i] + Ba[i] * np.cos(2 * np.pi * x)
        ax.plot(x, ya, c="red")

def popDynamics(rawData, leg=False):
    n = int(len(rawData.index)/2)

    j_data = rawData.iloc[:n]
    a_data = rawData.iloc[n:2*n]
    F_data = rawData.loc['F']

    fig1, ax1 = plt.subplots()
    j_data.T.plot(ax=ax1, title="Молодые особи", xlabel="t", legend=leg)
    aj_yMax = j_data.max().max()

    fig2, ax2 = plt.subplots()
    a_data.T.plot(ax=ax2, title="Взрослые особи", xlabel="t", legend=leg)
    a_yMax = a_data.max().max()
    if (a_yMax > aj_yMax):
        aj_yMax = a_yMax

    fig3, ax3 = plt.subplots()
    F_data.T.plot(ax=ax3, title="Хищник", xlabel="t", legend=leg)

    ax1.set_ylim([0, aj_yMax*1.1])
    ax2.set_ylim([0, aj_yMax*1.1])
    ax3.set_ylim([0, F_data.max()*1.1])
    # # # ax1.set_xlim([-5, 400])
    # # # ax2.set_xlim([-5, 400])
    # # # ax3.set_xlim([-5, 400])
    # ax1.set_xscale('log')
    # ax2.set_xscale('log')
    # ax3.set_xscale('log')
    # # ax1.set_xlim([0.2, ax1.get_xlim()[1]])
    # # ax2.set_xlim([0.2, ax2.get_xlim()[1]])
    # # ax3.set_xlim([0.2, ax3.get_xlim()[1]])
    # # # ax1.set_xscale('log', base=2.71828)
    # # # ax2.set_xscale('log', base=2.71828)
    # # # ax3.set_xscale('log', base=2.71828)
    # # #     for ax in [ax1, ax2, ax3]:
    # # #         ticks = ax.get_xticks()
    # # #         print(ticks)
    # # #         _ticks = []
    # # #         for tick in ticks:
    # # #             tmp = tick
    # # #             _tick = 0
    # # #             while (tmp != 1.0):
    # # #                 tmp /= np.e
    # # #                 _tick += 1
    # # #                 if (_tick > 100):
    # # #                     break
    # # #             if (_tick > 100):
    # # #                 tmp = tick
    # # #                 _tick = 0
    # # #                 while (tmp != 1.0):
    # # #                     tmp *= np.e
    # # #                     _tick -= 1
    # # #                     if (_tick < -100):
    # # #                         break
    # # #             _ticks.append(_tick)
    # # #         print(_ticks)
    # # #         ax.set_xticks(_ticks)
    # # #         ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    # # #         ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

def histStrats(stratData):
    ax = stratData[['Aj','Bj','Aa','Ba']].hist(layout=(2, 2), figsize=(12, 6), bins=200)
    ax[0][0].figure.tight_layout()

def histMps(mpData):
    mpData.loc[:,'M1':'M8'].hist(layout=(2, 4), figsize=(12, 6), bins=200)[0][0].figure.tight_layout()

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

def corrMps_2(mpData):
    corrMatr=mpData.loc[:,'M1':'M8'].corr()

    fig, ax = plt.subplots()
    sns.heatmap(corrMatr, square=True, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
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
    s = ax.scatter(x_x, x_y, c=y, cmap=cMap, s=5, alpha=0.03)

    x_visual = np.linspace(-1,1)
    y_visual = -(lam1/lam2)*x_visual - lam0/lam2
    ax.plot(x_visual, y_visual, color="navy", label="ML")
    
    leg1 = ax.legend(*s.legend_elements(alpha=1), loc="lower left", title="Class", draggable=True)
    ax.add_artist(leg1)  # needs for ax.plot legend
    ax.legend(loc="upper right", title="Hyperplane", draggable=True)  # needs for ax.plot legend

    fig.tight_layout()

def clf3dPlane(selData, lams, M1, M2, M3, elevation=30, azimuth=-60, a=0.02):
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
    s = ax.scatter(x_x, x_y, x_z, c=y, cmap=cMap, alpha=a)
    #s = ax.scatter(x_x, x_y, x_z, c=y, cmap=cMap, alpha=0.5, s=1)

    tmp = np.linspace(-1,1)
    x_visual, y_visual = np.meshgrid(tmp,tmp)
    z_visual = lambda x_vis,y_vis: -(lam1/lam3)*x_vis - (lam2/lam3)*y_vis - lam0/lam3
    ax.plot_surface(x_visual, y_visual, z_visual(x_visual, y_visual), color="navy", alpha=0.5)
    ##ax.plot_wireframe(x_visual, y_visual, z_visual(x_visual, y_visual), rstride=5, cstride=5, color="navy", alpha=0.5)

    # xticks = ax.xaxis.get_major_ticks()
    # for i in range(len(xticks)):
    #     if (i%2 == 0):
    #         xticks[i].set_visible(False)
    # yticks = ax.yaxis.get_major_ticks()
    # for i in range(len(yticks)):
    #     if (i%2 == 0):
    #         yticks[i].set_visible(False)

    ax.view_init(elevation, azimuth)
    ax.legend(*s.legend_elements(alpha=1), title="Class")
    fig.tight_layout()

def clfPlanes(selData, lams):
    X = selData.loc[:,'M1':'M8'].values
    y = selData['class'].values

    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):  # (i,j)<->(y,x), тогда: по строкам - i, по столбцам - j
            ax[i][j].set(xlim=(-1, 1), ylim=(-1, 1))  # TODO: [i][j] ---> [i, j] more efficient (estimate it)
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
                ax[i][j].scatter(X[:, j], X[:, i], c=y, cmap=cMap, s=5, alpha=0.03)

                x_visual = np.linspace(-1, 1)
                y_visual = -(lams[j+1]/lams[i+1])*x_visual - lams[0]/lams[i+1]
                ax[i][j].plot(x_visual, y_visual, color="navy")

# в отличии от исх.свертки макропар-ов ур-е восст.гиперплоскости скорее всего содержит lam0!=0,
# если точность класс-ра не 100% и ему не сообщили считать lam0=0:
#   lam0 + lam1*M1 + lam2*M2 + lam3*M3 + ... + lam44*M8M8 = 0  ||  lam1*M1 + lam2*M2 + lam3*M3 + ... + lam44*M8M8 = b
# lam0 следует исп-ть в совокуп-ти с норм.разностями при демонстрации рез-та обучения на проекциях, но не в дальнейшем!

# пусть:
#   w := (lam1,lam2,...,lam44)^T, x := (M1,M2,...,M8M8), b := -lam0
# тогда для двухмерной проекции:
#   <w,x> - b = 0 ---> w^T * x - b = 0 ---> (lam1,lam2) * (M1, M2)^T - b = 0
                                            # ---> lam1*M1 + lam2*M2 + lam0 = 0 ---> M2 = -(lam1/lam2)*M1 - lam0/lam2
# для трехмерной проекции итд:
#   lam1*M1 + lam2*M2 + lam3*M3 + ... + lam0 = 0 ---> ...
    
    fig.tight_layout()
    return fig

def phasePortrait(p, q, r, s):
    z1, z2, F = np.meshgrid(np.arange(-0.1, 1, 0.1), np.arange(-0.1, 1, 0.1), np.arange(-0.1, 1, 0.1))

    dz1_dt = -p*z1 - q*z1*F + r*z2 - z1*(z1 + z2)
    dz2_dt = p*z1 - s*z2*F - z2*(z1 + z2)
    dF_dt = (q*z1 + s*z2)*F - F
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.quiver(z1, z2, F, dz1_dt, dz2_dt, dF_dt, length=0.1, normalize=True)
    ax.quiver(z1, z2, F, dz1_dt, dz2_dt, dF_dt, length=0.1, normalize=True)
    fig.tight_layout()

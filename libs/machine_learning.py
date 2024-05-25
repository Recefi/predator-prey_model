from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tqdm
from joblib import Parallel, delayed
import seaborn as sns


def runClfSVM(selData):
    """
    "For the linear case, the algorithm used in LinearSVC by the liblinear implementation is much more efficient
        than its libsvm-based SVC counterpart and can scale almost linearly to millions of samples and/or features."
    "LinearSVC is less sensitive to C when it becomes large,
        and prediction results stop improving after a certain threshold. 
            Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer."
    """
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # X_train.loc[:,'M7'].plot(kind='box')
    # plt.show()
    # print(X_train['M7'].quantile([0.001,.005,.01,.04,.045,.05,.1,.2,.5,.9,.91,.925,.95,.99,.995,.999]))
    # indexes = X_train[(X_train['M7']<X_train['M7'].quantile(.02)) | (X_train['M7']>X_train['M7'].quantile(.98))].index
    # X_train = X_train.drop(index=indexes)
    # y_train = y_train.drop(index=indexes)
    # X_train.loc[:,'M7'].plot(kind='box')
    # plt.show()
    # corrMatr = X_train.loc[:, 'M2M2':'M6M6'].corr()
    # sns.heatmap(corrMatr, square=True, annot=True, fmt='.2f', vmin=-1, vmax=1, cmap='coolwarm')
    # plt.show()

    # Нормирование
    mpMaxsSeries = X_train.loc[:,'M1':'M8M8'].abs().max()
    #mpMaxsSeries = X_train.loc[:,'M1':'M8M8'].std()  # или стандартизация
    X_train.loc[:,'M1':'M8M8'] = X_train.loc[:,'M1':'M8M8'] / mpMaxsSeries
    X_test.loc[:,'M1':'M8M8'] = X_test.loc[:,'M1':'M8M8'] / mpMaxsSeries

    # проверка, что выборка действительно центрирована как по значениям фетчей,
    #                                                   так и по меткам класса, чтобы использовать fit_intercept=False
    print("sel size: ", len(y))
    print("train sel size: ", len(y_train))
    print("sel mean:")
    print(pd.DataFrame({"orig_X": selData.loc[:,'M1':'M8M8'].mean(), "orig_y": y.mean(),
                                                        "train_X": X_train.mean(axis=0), "train_y": y_train.mean()}))

    # Train
    clf = svm.LinearSVC(fit_intercept=False, dual=False, random_state=55, C=1)
    #clf = svm.SVC(kernel="linear", random_state=55, C=1)
    clf.fit(X_train, y_train)
    lams = clf.coef_[0]
    print(lams)

    # def mlIter(i):
    #     clf = svm.LinearSVC(fit_intercept=False, dual=False, random_state=55, C=i*1e-2)
    #     clf.fit(X_train, y_train)
    #     return clf

    if (lams[38] > 0 and lams[0] > 0 and lams[1] > 0 and lams[2] > 0 and lams[3] > 0
                        and lams[4] > 0 and lams[5] > 0 and lams[6] > 0 and lams[7] > 0):
        print("Ok")
    # else:
    #     Cs = []
    #     res = Parallel(n_jobs=-1)(delayed(mlIter)(i) for i in tqdm.tqdm(range(100, 1001)))
    #     for _clf in tqdm.tqdm(res):
    #         _lams = _clf.coef_[0]
    #         if (_lams[38] > 0 and _lams[0] > 0 and _lams[1] > 0 and _lams[2] > 0 and _lams[3] > 0
    #                             and _lams[4] > 0 and _lams[5] > 0 and _lams[6] > 0 and _lams[7] > 0):
    #             Cs.append(_clf.get_params()['C'])
    #             clf = _clf
    #             lams = _lams
    #     if not Cs:
    #         res = Parallel(n_jobs=-1)(delayed(mlIter)(i) for i in tqdm.tqdm(range(100, 0, -1)))
    #         for _clf in tqdm.tqdm(res):
    #             _lams = _clf.coef_[0]
    #             if (_lams[38] > 0 and _lams[0] > 0 and _lams[1] > 0 and _lams[2] > 0 and _lams[3] > 0
    #                                 and _lams[4] > 0 and _lams[5] > 0 and _lams[6] > 0 and _lams[7] > 0):
    #                 Cs.append(_clf.get_params()['C'])
    #                 clf = _clf
    #                 lams = _lams
    #     print(Cs)

    # Test
    print(confusion_matrix(y_test, clf.predict(X_test)))
    print("Точность классификатора на обучающей выборке:", (clf.predict(X_train) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (clf.predict(X_test) == y_test).mean()*100)

    if (clf.intercept_): lams = np.concatenate([clf.intercept_, lams])
    else: lams = np.concatenate([[clf.intercept_], lams])

    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print ("ml time: ", time.time() - start)
    return lams, mpMaxsData

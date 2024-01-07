from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


def runClfSVM(selData):
    """
    "For the linear case, the algorithm used in LinearSVC by the liblinear implementation is much more efficient
        than its libsvm-based SVC counterpart and can scale almost linearly to millions of samples and/or features."
    "LinearSVC is less sensitive to C when it becomes large, and prediction results stop improving after a certain threshold. 
        Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer."
    """
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.loc[:,'M1':'M8M8'].abs().max()
    X_train.loc[:,'M1':'M8M8'] = X_train.loc[:,'M1':'M8M8'] / mpMaxsSeries
    X_test.loc[:,'M1':'M8M8'] = X_test.loc[:,'M1':'M8M8'] / mpMaxsSeries

    # выборка должна быть центрированной как по значениям фетчей, так и по меткам класса, чтобы использовать fit_intercept=False
    print("sel size: ", len(y))
    print("train sel size: ", len(y_train))
    print("sel mean:")
    print(pd.DataFrame({"orig_X": selData.loc[:,'M1':'M8M8'].mean(), "orig_y": y.mean(), "train_X": X_train.mean(axis=0), "train_y": y_train.mean()}))

    # Train
    clf = svm.LinearSVC(fit_intercept=False, dual=False, random_state=55)
    clf.fit(X_train, y_train)   

    # Test
    print(confusion_matrix(y_test, clf.predict(X_test)))
    print("Точность классификатора на обучающей выборке:", (clf.predict(X_train) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (clf.predict(X_test) == y_test).mean()*100)

    lams = clf.coef_[0]
    if (clf.intercept_): lams = np.concatenate([clf.intercept_, lams])
    else: lams = np.concatenate([[clf.intercept_], lams])

    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print ("ml time: ", time.time() - start)
    return lams, mpMaxsData

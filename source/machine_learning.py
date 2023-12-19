from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    X = selData.loc[:,'M1':'M8M8'].values
    y = selData['class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    print("sel size: ", len(y))
    print("train sel size: ", len(y_train))
    print("-1:", len(y_train[y_train == -1]))
    print("1:", len(y_train[y_train == 1]))
    print("sel mean:")
    print(pd.DataFrame({"orig": selData.loc[:,'M1':'M8M8'].mean(), "train": X_train.mean(axis=0)}))

    # Train
    clf = svm.LinearSVC(fit_intercept=False, dual=False)
    clf.fit(X_train, y_train)

    # Test
    y_preds = clf.predict(X_test)
    print("Точность классификатора:", accuracy_score(y_test, y_preds)*100)

    lams = clf.coef_[0]
    if (clf.intercept_): lams = np.concatenate([clf.intercept_, lams])
    else: lams = np.concatenate([[clf.intercept_], lams])

    print("lam0 =",lams[0])
    print(lams)

    end = time.time()
    print ("ml time: ", end - start)
    return lams

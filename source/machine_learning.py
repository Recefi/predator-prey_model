from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    print("sel size: ", len(y))
    print("-1:", selData[selData['class'] == -1]['class'].count())
    print("1:", selData[selData['class'] == 1]['class'].count())
    print("sel mean:", selData.loc[:,'M1':'M8M8'].mean(), sep='\n')

    # Train
    clf = svm.LinearSVC(fit_intercept=False, dual="auto")
    clf.fit(X_train, y_train)

    # Test
    y_preds = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(y_test, y_preds)*100)

    lams = clf.coef_[0]
    if (clf.intercept_): lam0 = clf.intercept_[0]
    else: lam0 = clf.intercept_
    print("lam0 =",lam0)

    end = time.time()
    print ("ml time: ", end - start)
    return lams, lam0

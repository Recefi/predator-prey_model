from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np


def runClfSVM1(selData):
    """
    "The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples
        and may be impractical beyond a couple of 10000 samples. For large datasets consider using LinearSVC or SGDClassifier instead, 
            possibly [for non-linear kernel] after a Nystroem transformer or other Kernel Approximation."
    """
    X = selData.loc[:,'M1':'M8M8'].values
    y = selData['class'].values
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Train
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)

    # Test
    y_preds = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(y_test, y_preds)*100)

    lams = clf.coef_[0]
    intercept = clf.intercept_[0]

    return lams, intercept

def runClfSVM2(selData):
    """
    "For the linear case, the algorithm used in LinearSVC by the liblinear implementation is much more efficient
        than its libsvm-based SVC counterpart and can scale almost linearly to millions of samples and/or features."
    "LinearSVC is less sensitive to C when it becomes large, and prediction results stop improving after a certain threshold. 
        Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer."
    """
    X = selData.loc[:,'M1':'M8M8'].values
    y = selData['class'].values
    print(len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Train
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)

    # Test
    y_preds = clf.predict(X_test)
    print('Точность классификатора:')
    print('     SVM: ', accuracy_score(y_test, y_preds)*100)

    lams = clf.coef_[0]
    intercept = clf.intercept_[0]

    return lams, intercept

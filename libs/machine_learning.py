from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tqdm
from joblib import Parallel, delayed
from numba import jit, njit, prange

import libs.gen_selection as gs
import libs.taylor as tr
import libs.utility as ut

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')


def checkSelData_1(tmpSelData, act_y, pred_y, name):
    tmpSelData.insert(0, column='Actual label', value=act_y)
    tmpSelData.insert(1, column='Predicted label', value=pred_y)
    ut.writeData(tmpSelData, name+"_sel_data", callerName="dynamic_pred", subDirsName="checkMl")
    tmpSelData.to_excel("csv/dynamic_pred/checkMl/"+name+"_sel_data.xlsx")

def checkSelData_2(tmpSelData, pred_y, name):
    tmpSelData['Predicted label'] = pred_y
    ut.writeData(tmpSelData, name+"_sel_data", callerName="dynamic_pred", subDirsName="checkMl")
    tmpSelData.to_excel("csv/dynamic_pred/checkMl/"+name+"_sel_data.xlsx")

def drawCM(act_y, pred_y, name, classes):
    CM = confusion_matrix(act_y, pred_y)
    ConfusionMatrixDisplay(CM, display_labels=classes).plot(cmap = 'Reds').ax_.set_title(name.title())

def predictByLams(X_data, lams, neg_class=-1):
    diffMpMatr = X_data.loc[:, 'M1':'M8M8'].values
    diffFits = gs.calcLinsum(diffMpMatr, lams)
    y_pred = np.zeros(len(diffFits))
    for i in range(len(diffFits)):
        if diffFits[i] > 0:
            y_pred[i] = 1
        elif diffFits[i] < 0:
            y_pred[i] = neg_class
        else:
            print("WARNING: fits are equal?!")
    return y_pred

@njit
def predictByLamsOpt(diffMpMatr, lams):
    diffFits = gs.calcLinsum(diffMpMatr, lams)
    y_pred = np.zeros(len(diffFits))
    for i in range(len(diffFits)):
        if diffFits[i] > 0:
            y_pred[i] = 1
        elif diffFits[i] < 0:
            y_pred[i] = -1
        else:
            print("WARNING: fits are equal?!")
    return y_pred

def runClfSVM(selData):
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.abs().max()
    # или стандартизация
    #mpMaxsSeries = X_train.std()
    X_train = X_train / mpMaxsSeries
    X_test = X_test / mpMaxsSeries

    # проверка, что выборка действительно центрирована как по значениям фетчей,
    #                                                   так и по меткам класса, чтобы использовать fit_intercept=False
    print("sel size: ", len(y))
    print("train sel size: ", len(y_train))
    print("sel mean:")
    print(pd.DataFrame({"orig_X": selData.loc[:,'M1':'M8M8'].mean(), "orig_y": y.mean(),
                                                        "train_X": X_train.mean(axis=0), "train_y": y_train.mean()}))

    # Train
    #"For the linear case, the algorithm used in LinearSVC by the liblinear implementation is much more efficient
    #    than its libsvm-based SVC counterpart and can scale almost linearly to millions of samples and/or features."
    #"LinearSVC is less sensitive to C when it becomes large,
    #    and prediction results stop improving after a certain threshold. 
    #        Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer."
    clf = svm.LinearSVC(fit_intercept=False, dual=False, random_state=55, C=1)
    #clf = svm.SVC(kernel="linear", random_state=55, C=1)
    clf.fit(X_train, y_train)
    lams = clf.coef_[0]
    print(lams)

    if (lams[38] > 0 and lams[0] > 0 and lams[1] > 0 and lams[2] > 0 and lams[3] > 0
                        and lams[4] > 0 and lams[5] > 0 and lams[6] > 0 and lams[7] > 0):
        print("Ok")

    # Test
    print("Точность классификатора на обучающей выборке:", (clf.predict(X_train) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (clf.predict(X_test) == y_test).mean()*100)
    print("Точность классификатора на всей выборке:", (clf.predict(X / mpMaxsSeries) == y).mean()*100)
    # drawCM(y_train, clf.predict(X_train), "train", clf.classes_)
    # drawCM(y_test, clf.predict(X_test), "test", clf.classes_)
    # drawCM(y, clf.predict(X / mpMaxsSeries), "all", clf.classes_)
    # plt.show()
    # tmpSelData = X_train.copy()
    # checkSelData_1(tmpSelData, y_train, clf.predict(X_train), "train")
    # tmpSelData = X_test.copy()
    # checkSelData_1(tmpSelData, y_test, clf.predict(X_test), "test")
    # tmpSelData = (X / mpMaxsSeries).copy()
    # checkSelData_1(tmpSelData, y, clf.predict(X / mpMaxsSeries), "all")

    if (clf.intercept_):
        lams = np.concatenate([clf.intercept_, lams])
    else:
        lams = np.concatenate([[clf.intercept_], lams])

    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print("ml time: ", time.time() - start)
    return lams, mpMaxsData

def runClfTF(selData):
    rand_seed = np.random.randint(100000)
    tf.keras.utils.set_random_seed(47)
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class'].replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.abs().max()
    # или стандартизация
    #mpMaxsSeries = X_train.std()
    X_train = X_train / mpMaxsSeries
    X_test = X_test / mpMaxsSeries

    # Train
    input_layer = tf.keras.Input(shape=(44,))
    dense_layer = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100  # 100 - 0.96  # 10000 - 0.98
    #, verbose=0
    )

    lams = model.layers[1].get_weights()[0].flatten()
    print(lams)

    # print(model.predict(X_train))
    # print(model.predict(X_train).flatten())
    # print(np.rint(model.predict(X_train).flatten()))

    # Test
    print("Точность классификатора на обучающей выборке:", (np.rint(model.predict(X_train).flatten()) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (np.rint(model.predict(X_test).flatten()) == y_test).mean()*100)
    print("Точность классификатора на всей выборке:", (np.rint(model.predict(X / mpMaxsSeries).flatten()) == y).mean()*100)
    print("Проверка точности на обучающей выборке:", (predictByLams(X_train, lams, neg_class=0) == y_train).mean()*100)

    lams = np.insert(lams, 0, 0)
    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print("ml time: ", time.time() - start)
    tf.keras.utils.set_random_seed(rand_seed)
    return lams, mpMaxsData

class CustomLayer_2ls(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer_2ls, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a_j = self.add_weight()
        self.b_j = self.add_weight()
        self.g_j = self.add_weight()
        self.d_j = self.add_weight()
        self.a_a = self.add_weight()
        self.b_a = self.add_weight()
        self.g_a = self.add_weight()
        self.d_a = self.add_weight()

    def call(self, inputs):
        (M1, M2, M3, M4, M5, M6, M7, M8, M11, M12, M13, M14, M15, M16, M17, M18,
        M22, M23, M24, M25, M26, M27, M28, M33, M34, M35, M36, M37, M38, M44, M45, M46, M47, M48,
        M55, M56, M57, M58, M66, M67, M68, M77, M78, M88) = tf.unstack(inputs[:, :], axis=1)

        a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = self.a_j, self.b_j, self.g_j, self.d_j, self.a_a, self.b_a, self.g_a, self.d_a

        p = a_j*M1 + b_j*M3 + d_j*M4
        q = -g_j*M2
        r = a_a*M5 + b_a*M7 + d_a*M8
        s = -g_a*M6
        pp = 1/2*(a_j*a_j*M11 + b_j*b_j*M33 + d_j*d_j*M44 + 2*a_j*b_j*M13 + 2*a_j*d_j*M14 + 2*b_j*d_j*M34)
        pq = -(a_j*g_j*M12 + b_j*g_j*M23 + d_j*g_j*M24)
        pr = a_j*a_a*M15 + a_j*b_a*M17 + a_j*d_a*M18 + b_j*a_a*M35 + b_j*b_a*M37 + b_j*d_a*M38 + d_j*a_a*M45 + d_j*b_a*M47 + d_j*d_a*M48
        ps = -(a_j*g_a*M16 + b_j*g_a*M36 + d_j*g_a*M46)
        qq = 1/2*g_j*g_j*M22
        qr = -(a_a*g_j*M25 + b_a*g_j*M27 + d_a*g_j*M28)
        qs = g_j*g_a*M26
        rr = 1/2*(a_a*a_a*M55 + b_a*b_a*M77 + d_a*d_a*M88 + 2*a_a*b_a*M57 + 2*a_a*d_a*M58 + 2*b_a*d_a*M78)
        rs = -(a_a*g_a*M56 + b_a*g_a*M67 + d_a*g_a*M68)
        ss = 1/2*g_a*g_a*M66

        return tf.stack([p, q, r, s, pp, pq, pr, ps, qq, qr, qs, rr, rs, ss], axis=1)

def runClfTF_tr_2ls(selData, epochs=1):
    rand_seed = np.random.randint(100000)
    tf.keras.utils.set_random_seed(47)
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class'].replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.abs().max()
    # или стандартизация
    #mpMaxsSeries = X_train.std()
    X_train = X_train / mpMaxsSeries
    X_test = X_test / mpMaxsSeries

    # Train
    input_layer = tf.keras.Input(shape=(44,))
    custom_layer = CustomLayer_2ls()(input_layer)
    dense_layer = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False)(custom_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)

    def custom_loss(y_true, y_pred):
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = model.layers[1].trainable_weights

        penalty = 0.0
        for weight in weights[:8]:
            penalty += tf.maximum(0.0, -weight)  #tf.reduce_sum(tf.maximum(0.0, -weight)) --- если многомерный weight
        total_loss = base_loss + penalty
        return total_loss

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs
    #, verbose=0
    )

    a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = model.layers[1].get_weights()
    print("params:", np.array((a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    h = model.layers[2].get_weights()[0].flatten()
    print("h:", h)
    lams = np.array(tr.calcCoefs(h, (a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    print(lams)

    # Test
    print("Точность классификатора на обучающей выборке:", (np.rint(model.predict(X_train).flatten()) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (np.rint(model.predict(X_test).flatten()) == y_test).mean()*100)
    print("Точность классификатора на всей выборке:", (np.rint(model.predict(X / mpMaxsSeries).flatten()) == y).mean()*100)
    print("Проверка точности на обучающей выборке:", (predictByLams(X_train, lams, neg_class=0) == y_train).mean()*100)

    lams = np.insert(lams, 0, 0)
    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print("ml time: ", time.time() - start)
    tf.keras.utils.set_random_seed(rand_seed)
    return lams, mpMaxsData

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a_j = self.add_weight()
        self.b_j = self.add_weight()
        self.g_j = self.add_weight()
        self.d_j = self.add_weight()
        self.a_a = self.add_weight()
        self.b_a = self.add_weight()
        self.g_a = self.add_weight()
        self.d_a = self.add_weight()

        self.hp = self.add_weight()
        self.hq = self.add_weight()
        self.hr = self.add_weight()
        self.hs = self.add_weight()
        self.hpp = self.add_weight()
        self.hpq = self.add_weight()
        self.hpr = self.add_weight()
        self.hps = self.add_weight()
        self.hqq = self.add_weight()
        self.hqr = self.add_weight()
        self.hqs = self.add_weight()
        self.hrr = self.add_weight()
        self.hrs = self.add_weight()
        self.hss = self.add_weight()

    def call(self, inputs):
        (M1, M2, M3, M4, M5, M6, M7, M8, M11, M12, M13, M14, M15, M16, M17, M18,
        M22, M23, M24, M25, M26, M27, M28, M33, M34, M35, M36, M37, M38, M44, M45, M46, M47, M48,
        M55, M56, M57, M58, M66, M67, M68, M77, M78, M88) = tf.unstack(inputs[:, :], axis=1)

        a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = self.a_j, self.b_j, self.g_j, self.d_j, self.a_a, self.b_a, self.g_a, self.d_a

        p = a_j*M1 + b_j*M3 + d_j*M4
        q = -g_j*M2
        r = a_a*M5 + b_a*M7 + d_a*M8
        s = -g_a*M6
        pp = 1/2*(a_j*a_j*M11 + b_j*b_j*M33 + d_j*d_j*M44 + 2*a_j*b_j*M13 + 2*a_j*d_j*M14 + 2*b_j*d_j*M34)
        pq = -(a_j*g_j*M12 + b_j*g_j*M23 + d_j*g_j*M24)
        pr = a_j*a_a*M15 + a_j*b_a*M17 + a_j*d_a*M18 + b_j*a_a*M35 + b_j*b_a*M37 + b_j*d_a*M38 + d_j*a_a*M45 + d_j*b_a*M47 + d_j*d_a*M48
        ps = -(a_j*g_a*M16 + b_j*g_a*M36 + d_j*g_a*M46)
        qq = 1/2*g_j*g_j*M22
        qr = -(a_a*g_j*M25 + b_a*g_j*M27 + d_a*g_j*M28)
        qs = g_j*g_a*M26
        rr = 1/2*(a_a*a_a*M55 + b_a*b_a*M77 + d_a*d_a*M88 + 2*a_a*b_a*M57 + 2*a_a*d_a*M58 + 2*b_a*d_a*M78)
        rs = -(a_a*g_a*M56 + b_a*g_a*M67 + d_a*g_a*M68)
        ss = 1/2*g_a*g_a*M66

        sum = self.hp*p + self.hq*q + self.hr*r + self.hs*s + self.hpp*pp + self.hpq*pq + self.hpr*pr \
                    + self.hps*ps + self.hqq*qq + self.hqr*qr + self.hqs*qs + self.hrr*rr + self.hrs*rs + self.hss*ss
        return tf.nn.sigmoid(tf.reshape(sum, (-1, 1)))  # tf.reshape: (None, ) ---> (None, 1)

def runClfTF_tr(selData, epochs=1):
    rand_seed = np.random.randint(100000)
    tf.keras.utils.set_random_seed(47)
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class'].replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.abs().max()
    # или стандартизация
    #mpMaxsSeries = X_train.std()
    X_train = X_train / mpMaxsSeries
    X_test = X_test / mpMaxsSeries

    # Train
    input_layer = tf.keras.Input(shape=(44,))
    custom_layer = CustomLayer()(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=custom_layer)

    def custom_loss(y_true, y_pred):
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = model.layers[1].trainable_weights

        penalty = 0.0
        for weight in weights[:8]:
            penalty += tf.maximum(0.0, -weight)  #tf.reduce_sum(tf.maximum(0.0, -weight)) --- если многомерный weight
        total_loss = base_loss + penalty
        return total_loss

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs
    #, verbose=0
    )

    weights = model.layers[1].get_weights()
    a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = weights[:8]
    print("params:", np.array((a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss = weights[8:]
    print("h:", np.array((hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss)))
    lams = np.array(tr.calcCoefs((hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss),
                                (a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    print(lams)

    # Test
    print("Точность классификатора на обучающей выборке:", (np.rint(model.predict(X_train).flatten()) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (np.rint(model.predict(X_test).flatten()) == y_test).mean()*100)
    print("Точность классификатора на всей выборке:", (np.rint(model.predict(X / mpMaxsSeries).flatten()) == y).mean()*100)
    print("Проверка точности на обучающей выборке:", (predictByLams(X_train, lams, neg_class=0) == y_train).mean()*100)

    lams = np.insert(lams, 0, 0)
    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print("ml time: ", time.time() - start)
    tf.keras.utils.set_random_seed(rand_seed)
    return lams, mpMaxsData

class CustomLayer_F(tf.keras.layers.Layer):
    def __init__(self, F, **kwargs):
        super(CustomLayer_F, self).__init__(**kwargs)
        self.F = tf.constant(F, dtype=tf.float32)

    def build(self, input_shape):
        self.a_j = self.add_weight()
        self.b_j = self.add_weight()
        self.g_j = self.add_weight()
        self.d_j = self.add_weight()
        self.a_a = self.add_weight()
        self.b_a = self.add_weight()
        self.g_a = self.add_weight()
        self.d_a = self.add_weight()

        self.hp = self.add_weight()
        self.hq = self.add_weight()
        self.hr = self.add_weight()
        #self.hs = self.add_weight()
        self.hpp = self.add_weight()
        self.hpq = self.add_weight()
        self.hpr = self.add_weight()
        #self.hps = self.add_weight()
        self.hqq = self.add_weight()
        self.hqr = self.add_weight()
        #self.hqs = self.add_weight()
        self.hrr = self.add_weight()
        #self.hrs = self.add_weight()
        #self.hss = self.add_weight()

    def call(self, inputs):
        (M1, M2, M3, M4, M5, M6, M7, M8, M11, M12, M13, M14, M15, M16, M17, M18,
        M22, M23, M24, M25, M26, M27, M28, M33, M34, M35, M36, M37, M38, M44, M45, M46, M47, M48,
        M55, M56, M57, M58, M66, M67, M68, M77, M78, M88) = tf.unstack(inputs[:, :], axis=1)

        a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = self.a_j, self.b_j, self.g_j, self.d_j, self.a_a, self.b_a, self.g_a, self.d_a
        F = self.F

        p = a_j*M1 + b_j*M3 + d_j*M4
        q = -F*g_j*M2
        r = a_a*M5 + b_a*M7 + d_a*M8
        s = -F*g_a*M6
        pp = 1/2*(a_j*a_j*M11 + b_j*b_j*M33 + d_j*d_j*M44 + 2*a_j*b_j*M13 + 2*a_j*d_j*M14 + 2*b_j*d_j*M34)
        pq = -F*(a_j*g_j*M12 + b_j*g_j*M23 + d_j*g_j*M24)
        pr = a_j*a_a*M15 + a_j*b_a*M17 + a_j*d_a*M18 + b_j*a_a*M35 + b_j*b_a*M37 + b_j*d_a*M38 + d_j*a_a*M45 + d_j*b_a*M47 + d_j*d_a*M48
        ps = -F*(a_j*g_a*M16 + b_j*g_a*M36 + d_j*g_a*M46)
        qq = 1/2*F*F*g_j*g_j*M22
        qr = -F*(a_a*g_j*M25 + b_a*g_j*M27 + d_a*g_j*M28)
        qs = F*F*g_j*g_a*M26
        rr = 1/2*(a_a*a_a*M55 + b_a*b_a*M77 + d_a*d_a*M88 + 2*a_a*b_a*M57 + 2*a_a*d_a*M58 + 2*b_a*d_a*M78)
        rs = -F*(a_a*g_a*M56 + b_a*g_a*M67 + d_a*g_a*M68)
        ss = 1/2*F*F*g_a*g_a*M66

        hs = -2 - self.hq
        hqs = -self.hqq
        hss = self.hqq

        hps = -self.hpq
        hrs = -self.hqr

        sum = self.hp*p + self.hq*q + self.hr*r + hs*s + self.hpp*pp + self.hpq*pq + self.hpr*pr \
                    + hps*ps + self.hqq*qq + self.hqr*qr + hqs*qs + self.hrr*rr + hrs*rs + hss*ss
        return tf.nn.sigmoid(tf.reshape(sum, (-1, 1)))  # tf.reshape: (None, ) ---> (None, 1)

def runClfTF_tr_F(selData, FLim, epochs=1):
    rand_seed = np.random.randint(100000)
    tf.keras.utils.set_random_seed(47)
    start = time.time()
    X = selData.loc[:,'M1':'M8M8']
    y = selData['class'].replace(-1, 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=74,
                                                        train_size=4*(len(y)//5), shuffle=False)

    # Нормирование
    mpMaxsSeries = X_train.abs().max()
    # или стандартизация
    #mpMaxsSeries = X_train.std()
    X_train = X_train / mpMaxsSeries
    X_test = X_test / mpMaxsSeries

    # Train
    input_layer = tf.keras.Input(shape=(44,))
    custom_layer = CustomLayer_F(F=FLim)(input_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=custom_layer)

    def custom_loss(y_true, y_pred):
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        weights = model.layers[1].trainable_weights

        penalty = 0.0
        for weight in weights[:8]:
            penalty += tf.maximum(0.0, -weight)  #tf.reduce_sum(tf.maximum(0.0, -weight)) --- если многомерный weight
        total_loss = base_loss + penalty
        return total_loss

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs
    #, verbose=0
    )

    weights = model.layers[1].get_weights()
    a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a = weights[:8]
    print("params:", np.array((a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    hp, hq, hr, hpp, hpq, hpr, hqq, hqr, hrr = weights[8:]
    hs = -2 - hq
    hqs = -hqq
    hss = hqq
    hps = -hpq
    hrs = -hqr
    h = tr.imputeDers_qFsF((hp, hq, hr, hs, hpp, hpq, hpr, hps, hqq, hqr, hqs, hrr, hrs, hss), FLim)
    print("h:", np.array(h))
    lams = np.array(tr.calcCoefs(h, (a_j, b_j, g_j, d_j, a_a, b_a, g_a, d_a)))
    print(lams)

    # Test
    print("Точность классификатора на обучающей выборке:", (np.rint(model.predict(X_train).flatten()) == y_train).mean()*100)
    print("Точность классификатора на тестовой выборке:", (np.rint(model.predict(X_test).flatten()) == y_test).mean()*100)
    print("Точность классификатора на всей выборке:", (np.rint(model.predict(X / mpMaxsSeries).flatten()) == y).mean()*100)
    print("Проверка точности на обучающей выборке:", (predictByLams(X_train, lams, neg_class=0) == y_train).mean()*100)

    lams = np.insert(lams, 0, 0)
    print("lam0 =",lams[0])
    print(lams)

    mpMaxsData = pd.DataFrame(mpMaxsSeries).T
    print("ml time: ", time.time() - start)
    tf.keras.utils.set_random_seed(rand_seed)
    return lams, mpMaxsData

def predictByRstdDynamic(initStratPopData, stratPopData):
    initIdxs = initStratPopData.index.to_numpy(copy=True)
    n = len(initIdxs)
    idxs = stratPopData.index
    t = stratPopData['t']

    def assignClass(i, j):
        if (t[i] == t[j]):
            raise Exception("nullified at once")
        else:
            if (t[i] > t[j]):
                elem = 1
            else:
                elem = -1
        return elem

    _idxs = []
    j = 0
    for i in range(len(idxs)):
        while(j < n):
            if (initIdxs[j] == idxs[i]):
                _idxs.append(idxs[i])
                j+=1
                break
            elif (initIdxs[j] > idxs[i]):
                break
            j+=1
    print(initIdxs)
    print(idxs)
    print(_idxs)
    print("_strats: ", len(_idxs))

    dropCount = 0
    for i in range(n):
        if (initIdxs[i] != _idxs[i-dropCount]):
            dropCount+=1
            initIdxs[i] = -1
    print(dropCount)

    sel = []
    selIdxs = []
    nextSelIdx = 0
    for i in range(n):
        for j in range(i+1, n):
            if not (initIdxs[i] == -1 or initIdxs[j] == -1):
                elemClass = assignClass(initIdxs[i], initIdxs[j])
                sel.append(elemClass)
                sel.append(-elemClass)
                selIdxs.append(nextSelIdx)
                selIdxs.append(nextSelIdx+1)
            nextSelIdx+=2
    
    pred_y = pd.Series(sel, index=selIdxs)
    return pred_y

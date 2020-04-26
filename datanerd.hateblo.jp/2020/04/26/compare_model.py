# coding: utf-8
import os
import numpy as np
import pandas as pd
import random as rn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from plot_tool import plot, compress

rstate = 14

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

models = [
    RandomForestClassifier(n_estimators=300, max_leaf_nodes=32,
                           random_state=0),
    LogisticRegression(penalty="l1",
                       C=50.0,
                       solver="liblinear",
                       max_iter=150,
                       random_state=0),
    LogisticRegression(penalty="l1",
                       C=1.0,
                       solver="liblinear",
                       max_iter=150,
                       random_state=0),
]
max_features = [80, 35, 18]


def int2categorical(data, fname, max_num=None):
    out = []
    if max_num is None:
        max_num = max(data)
    initial_vector = [0 for _ in range(max_num + 10)]
    fnames = [fname + "_{}".format(i) for i in range(max_num + 10)]
    for x in data:
        v = initial_vector[:]
        v[int(x)] = 1
        out.append(v)
    return np.array(out), fnames, max_num


def sex2num(data):
    return [0 if x == "male" else 1 for x in data]


def name2bow(data, vectorizer=None, train=False):
    if train is False:
        return vectorizer.transform(data), vectorizer
    else:
        vectorizer = CountVectorizer(min_df=5).fit(data)
        return vectorizer.transform(data), vectorizer


def step1(X, ms=None, vect=None, train=False):
    if ms is None:
        ms = [None, None, None, None]
    bow, vect = name2bow(X["Name"], vect, train)
    X["Sex"] = sex2num(X["Sex"])
    pcls, pfns, ms[0] = int2categorical(X["Pclass"], "Pclass", ms[0])
    ssa, sfns, ms[1] = int2categorical(X["Siblings/Spouses Aboard"], "SSA",
                                       ms[1])
    pca, pcfns, ms[2] = int2categorical(X["Parents/Children Aboard"], "PCA",
                                        ms[2])
    sex, sxfnx, ms[3] = int2categorical(X["Sex"], "SEX", ms[3])
    cat_feats = np.hstack((pcls, ssa, pca, sex))
    cat_fns = pfns + sfns + pcfns + sxfnx
    X_fix = X.drop([
        "Name", "Pclass", "Sex", "Siblings/Spouses Aboard",
        "Parents/Children Aboard"
    ], 1)
    feats = np.hstack((cat_feats, X_fix))
    fns = cat_fns + X_fix.columns.tolist()
    poly = PolynomialFeatures(2).fit(feats)
    fnames1 = ["BOW_" + x for x in vect.get_feature_names()]
    fnames2 = poly.get_feature_names(fns)
    fnames3 = fnames1 + fnames2
    X_fix = np.hstack((bow.toarray(), poly.transform(feats)))
    return X_fix, vect, fnames3, ms


def step2(X, fnames, scaler_std=None, scaler_m=None, train=False):
    if train:
        scaler_std = StandardScaler().fit(X)
        scaler_m = MinMaxScaler().fit(X)
    X_std = scaler_std.transform(X)
    X_m = scaler_m.transform(X)
    if train:
        fnames0 = fnames[:]
        fnames1 = [x + "_s" for x in fnames0]
        fnames2 = [x + "_m" for x in fnames0]
        fnames3 = fnames0 + fnames1 + fnames2
    else:
        fnames3 = None
    return np.hstack((X, X_std, X_m)), scaler_std, scaler_m, fnames3


def step3(X, y, fns, sfms, train=False):
    X_fix = X[:]
    if fns is not None:
        fns_fix = np.array(fns[:])
    else:
        fns_fix = None
    if train:
        sfms = []
    for i, (m, mf) in enumerate(zip(models, max_features)):
        if train:
            m.fit(X_fix, y)
            sfm = SelectFromModel(m,
                                  threshold="median",
                                  prefit=True,
                                  max_features=mf)
            sfms.append(sfm)
            if fns is not None:
                mask = sfm.get_support(indices=True)
                fns_fix = fns_fix[mask]
        sfm = sfms[i]
        X_fix = sfm.transform(X_fix)
    return X_fix, fns_fix, sfms


def features(X,
             y,
             ms=None,
             vect=None,
             s_std=None,
             s_m=None,
             sfms=None,
             train=False):
    X_1, vect, tmp_fns, ms = step1(X, ms, vect, train)
    X_2, s_std, s_m, fns = step2(X_1, tmp_fns, s_std, s_m, train)
    X_3, fns, sfms = step3(X_2, y, fns, sfms, train)
    return X_3, ms, vect, s_std, s_m, sfms, fns


def features_keras(X, ms=None, vect=None, scaler=None, train=False):
    if ms is None:
        ms = [None, None, None, None]
    bow, vect = name2bow(X["Name"], vect, train)
    X["Sex"] = sex2num(X["Sex"])
    pcls, pfns, ms[0] = int2categorical(X["Pclass"], "Pclass", ms[0])
    ssa, sfns, ms[1] = int2categorical(X["Siblings/Spouses Aboard"], "SSA",
                                       ms[1])
    pca, pcfns, ms[2] = int2categorical(X["Parents/Children Aboard"], "PCA",
                                        ms[2])
    sex, sxfnx, ms[3] = int2categorical(X["Sex"], "SEX", ms[3])
    cat_feats = np.hstack((pcls, ssa, pca, sex))
    cat_fns = pfns + sfns + pcfns + sxfnx
    X_fix = X.drop([
        "Name", "Pclass", "Sex", "Siblings/Spouses Aboard",
        "Parents/Children Aboard"
    ], 1)
    feats = np.hstack((cat_feats, X_fix))
    fns = cat_fns + X_fix.columns.tolist()
    fnames1 = ["BOW_" + x for x in vect.get_feature_names()]
    fnames2 = fns
    fnames3 = fnames1 + fnames2
    X_fix = np.hstack((bow.toarray(), feats))
    if train:
        scaler = StandardScaler().fit(X_fix)
    X_fix = scaler.transform(X_fix)
    return X_fix, vect, fnames3, ms, scaler


def logreg_keras(input_dim, fe=True):
    model = Sequential()

    if fe:
        model.add(Dense(100, input_dim=input_dim, activation="tanh"))
        model.add(
            Dense(1,
                  activation='sigmoid',
                  kernel_regularizer=regularizers.l1(0.1)))
    else:
        model.add(
            Dense(1,
                  input_dim=input_dim,
                  activation='sigmoid',
                  kernel_regularizer=regularizers.l1(0.1)))
    model.compile(loss="binary_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])
    return model


def run_sklearn(X_train, X_test, y_train, y_test, plot_data=None, outname="test"):
    X_train_fx, ms, vect, s_std, s_m, sfms, fns = features(X_train,
                                                           y_train,
                                                           ms=None,
                                                           train=True)
    model = LogisticRegression(C=10.0, max_iter=2000)
    model.fit(X_train_fx, y_train)
    X_test_fx, *_ = features(X_test,
                             None,
                             ms,
                             vect,
                             s_std,
                             s_m,
                             sfms,
                             train=False)
    y_pred = model.predict(X_test_fx)
    if plot_data is None:
        raise Exception("no plot data")
    plot(plot_data, y_test, y_pred, outname)

    print(classification_report(y_test, y_pred))
    cs = sorted([(c, fn) for c, fn in zip(model.coef_[0], fns)],
                key=lambda x: abs(x[0]),
                reverse=True)
    for c, fn in cs:
        if c != 0.0:
            print("{}: {}".format(fn, c))
    print()


def run_keras(X_train, X_test, y_train, y_test, plot_data=None, fe=False, outname="test"):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:],
                                                        df.iloc[:, 0],
                                                        random_state=rstate)
    X_train_fx, vect, fns, ms, scaler = features_keras(X_train, train=True)
    model = logreg_keras(X_train_fx.shape[1], fe=fe)
    model.summary()
    model.fit(X_train_fx, y_train, epochs=10, batch_size=1, verbose=False)
    X_test_fx, *_ = features_keras(X_test, ms, vect, scaler, train=False)
    y_pred = model.predict_classes(X_test_fx)
    if plot_data is None:
        plot_data = compress(X_test_fx)
    plot(plot_data, y_test, y_pred, outname)
    print(classification_report(y_test, y_pred))
    return plot_data


if __name__ == "__main__":
    df = pd.read_csv("titanic.csv")
    data = train_test_split(df.iloc[:, 1:],
                            df.iloc[:, 0],
                            random_state=rstate)

    print("[keras(simple)]")
    plot_data = run_keras(*data, fe=False, outname="keras_simple")

    print("[sklearn]")
    run_sklearn(*data, plot_data, outname="sklearn")


    print("[keras(2 layers)]")
    run_keras(*data, plot_data, fe=True, outname="keras_2layers")

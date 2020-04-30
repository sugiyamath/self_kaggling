import sys
import os
import pandas as pd
import numpy as np
import random as rn
from subprocess import check_output
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

mln = 4

rstate = 6
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)


class MyEnsemble(BaseEstimator):
    def __init__(self, clss):
        self._clss = clss
        self._clfs = []

    def fit(self, X, y):
        for cls in self._clss:
            self._clfs.append(
                DecisionTreeClassifier(criterion="entropy",
                                       max_leaf_nodes=mln,
                                       random_state=rstate).fit(X[cls], y))
        return self

    def predict(self, X):
        preds = []
        for clf, cls in zip(self._clfs, self._clss):
            preds.append(clf.predict_proba(X[cls])[:, 1])
        y_pred = np.mean(preds, axis=0) > 0.5
        return y_pred

    def predict_proba(self, X):
        preds = []
        for clf, cls in zip(self._clfs, self._clss):
            preds.append(clf.predict_proba(X[cls]))
        y_pred = np.mean(preds, axis=0)
        return y_pred

    def get_params(self, deep=True):
        return {"clss": self._clss}

    def set_params(self, params):
        self._clss = params["clss"]


def _execute(X, y, clf, prefix, cls, ensemble=False, rf=False):
    if ensemble:
        prefixs = prefix
        clfs = clf._clfs
        clss = clf._clss
    else:
        prefixs = [prefix]
        clfs = [clf]
        clss = [cls]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=rstate)
    cv_result = cross_val_score(clf, X_train, y_train, cv=10, scoring="roc_auc")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(prefixs[0])
    print(cv_result)
    print("cv_mean:", np.mean(cv_result), ", cv_std:", np.std(cv_result))
    print(classification_report(y_test, y_pred))
    print()

    for cls, clf, prefix in zip(clss, clfs, prefixs):
        if not rf:
            export_graphviz(clf,
                            "./tree/{}.dot".format(prefix),
                            feature_names=cls,
                            filled=True)
            check_output("dot -Tpng ./tree/{}.dot -o ./tree/{}.png".format(
                prefix, prefix),
                         shell=True)


def run_model1(X, y):
    cls = X.columns.tolist()
    clf = DecisionTreeClassifier(criterion="entropy",
                                 random_state=rstate)
    _execute(X, y, clf, prefix="model1", cls=cls)


def run_model2(X, y):
    cls = X.columns.tolist()
    clf = DecisionTreeClassifier(criterion="entropy",
                                 max_leaf_nodes=mln,
                                 random_state=rstate)
    _execute(X, y, clf, prefix="model2", cls=cls)


def run_model3(X, y):
    cls = X.columns.tolist()
    clf = MyEnsemble([cls, cls])
    _execute(X,
             y,
             clf,
             prefix=["model3", "model3_B"],
             cls=cls,
             ensemble=True)


def run_model4(X, y):
    cls = X.columns.tolist()
    cls1 = ['Pclass', 'Sex', 'Age']
    cls2 = ['Age', 'Sex', 'Siblings/Spouses Aboard']
    clf = MyEnsemble([cls1, cls2])
    _execute(X,
             y,
             clf,
             prefix=["model4", "model4_B"],
             cls=cls,
             ensemble=True)


def _prepare_data(fname="./titanic.csv"):
    df = pd.read_csv(fname)
    del (df["Name"])
    df["Sex"] = [0 if x == "male" else 1 for x in df["Sex"]]
    return df.iloc[:, 1:], df.iloc[:, 0]



if __name__ == "__main__":
    data = _prepare_data()
    run_model1(*data)
    run_model2(*data)
    run_model3(*data)
    run_model4(*data)

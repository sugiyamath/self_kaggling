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

rstate = 6
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)

df = pd.read_csv("./titanic.csv")
del (df["Name"])
df["Sex"] = [0 if x == "male" else 1 for x in df["Sex"]]
X, y = df.iloc[:, 1:], df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rstate)


class MyEnsemble(BaseEstimator):
    def __init__(self, clss):
        self._clss = clss
        self._clfs = []

    def fit(self, X, y):
        for cls in self._clss:
            self._clfs.append(
                DecisionTreeClassifier(criterion="entropy",
                                       max_leaf_nodes=8,
                                       random_state=rstate).fit(X[cls], y))
        return self

    def predict(self, X):
        preds = []
        for clf, cls in zip(self._clfs, self._clss):
            preds.append(clf.predict_proba(X[cls])[:, 1])
        y_pred = np.mean(preds, axis=0) > 0.5
        return y_pred

    def get_params(self, deep=True):
        return {"clss": self._clss}

    def set_params(self, params):
        self._clss = params["clss"]


def _finalize(clf, y_pred, prefix, cls=X.columns, ensemble=False, rf=False):
    if ensemble:
        prefixs = prefix
        clfs = clf._clfs
        clss = clf._clss
    else:
        prefixs = [prefix]
        clfs = [clf]
        clss = [cls]

    print(prefix[0])
    print(cross_val_score(clf, X, y, cv=10, scoring="accuracy"))
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


def run_model1():
    clf = DecisionTreeClassifier(criterion="entropy",
                                 random_state=rstate).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    _finalize(clf, y_pred, "model1")


def run_model2():
    clf = DecisionTreeClassifier(criterion="entropy",
                                 max_leaf_nodes=8,
                                 random_state=rstate).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    _finalize(clf, y_pred, "model2")


def run_model3():
    clf = MyEnsemble([X.columns, X.columns]).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    _finalize(clf, y_pred, ["model3", "model3_B"], ensemble=True)


def run_model4():
    cls = X_train.columns.tolist()
    cls1 = ['Pclass', 'Sex', 'Age']
    cls2 = ['Age', 'Sex', 'Siblings/Spouses Aboard']
    clf = MyEnsemble([cls1, cls2]).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    _finalize(clf, y_pred, ["model4", "model4_B"], ensemble=True)


if __name__ == "__main__":
    run_model1()
    run_model2()
    run_model3()
    run_model4()

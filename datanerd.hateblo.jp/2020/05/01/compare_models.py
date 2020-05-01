import sys
import os
import pandas as pd
import numpy as np
import random as rn
from subprocess import check_output
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

mln = 8

rstate = 0
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(1)


class MyEnsemble(BaseEstimator):
    def __init__(self, clss):
        self._clss = clss
        self._clfs = []
        self._lr = None

    def fit(self, X, y):
        preds = []
        for cls in self._clss:
            clf = DecisionTreeClassifier(criterion="entropy",
                                         max_leaf_nodes=mln,
                                         random_state=rstate).fit(X[cls], y)
            preds.append(clf.predict_proba(X[cls])[:, 1:2])
            self._clfs.append(clf)
        feats = np.hstack(tuple(preds))
        self._lr = LogisticRegression().fit(feats, y)
        return self

    def predict(self, X):
        preds = []
        for clf, cls in zip(self._clfs, self._clss):
            preds.append(clf.predict_proba(X[cls])[:, 1:2])
        feats = np.hstack(tuple(preds))
        return self._lr.predict(feats)

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


def _execute(X, y, clf, prefix, cls, ensemble=False):
    alps = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if ensemble:
        prefixs = [prefix+"_{}".format(x) for x in alps]
        clfs = clf._clfs
        clss = clf._clss
    else:
        prefixs = [prefix]
        clfs = [clf]
        clss = [cls]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=rstate)
    cv_auc = cross_val_score(clf, X_train, y_train, cv=10, scoring="roc_auc")
    cv_acc = cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("[["+prefix+"]]")
    print("[cv_auc]")
    print(cv_auc)
    print("mean:", np.mean(cv_auc), ", std:", np.std(cv_auc))
    print()
    print("[cv_acc]")
    print(cv_acc)
    print("mean:", np.mean(cv_acc), ", std:", np.std(cv_acc))
    print()
    print(classification_report(y_test, y_pred))
    if ensemble:
        print("[model coefs]")
        for p, c in zip(prefixs, clf._lr.coef_[0]):
            print(p, c)
    print()

    for cls, clf, prefix in zip(clss, clfs, prefixs):
        print("["+prefix+" feature importances]")
        for n, fi in zip(cls, clf.feature_importances_):
            print(n, fi)
            export_graphviz(clf,
                            "./tree/{}.dot".format(prefix),
                            feature_names=cls,
                            filled=True)
            check_output("dot -Tpng ./tree/{}.dot -o ./tree/{}.png".format(
                prefix, prefix),
                         shell=True)
        print()
    print()
    print('-'*20)
    print()

def run_bad_model(X, y):
    X["rand1"] = [rn.randint(0, 10) for _ in range(X.shape[0])]
    X["rand2"] = [rn.randint(10, 20) for _ in range(X.shape[0])]
    X["rand3"] = [rn.randint(20, 30) for _ in range(X.shape[0])]
    cls = X.columns.tolist()
    clf = DecisionTreeClassifier(criterion="entropy", random_state=rstate)
    _execute(X, y, clf, prefix="bad_model", cls=cls)


def run_model1(X, y):
    cls = X.columns.tolist()
    clf = DecisionTreeClassifier(criterion="entropy", random_state=rstate)
    _execute(X, y, clf, prefix="model1", cls=cls)


def run_model2(X, y):
    cls = X.columns.tolist()
    clf = DecisionTreeClassifier(criterion="entropy",
                                 max_leaf_nodes=mln * 3,
                                 random_state=rstate)
    _execute(X, y, clf, prefix="model2", cls=cls)


def run_model3(X, y):
    cls = X.columns.tolist()
    clss = []
    clss.append(['Age', 'Sex', 'Pclass', 'Fare'])
    clss.append(['Age', 'Sex', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'])
    #clss.append(['Age', 'Sex', 'Fare'])
    clss.append(['Age', 'Sex'])
    #clss.append(['Age', 'Sex', 'Parents/Children Aboard'])
    clf = MyEnsemble(clss)
    _execute(X,
             y,
             clf,
             prefix="model3",
             cls=cls,
             ensemble=True)


def _prepare_data(fname="./titanic.csv"):
    df = pd.read_csv(fname)
    del (df["Name"])
    df["Sex"] = [0 if x == "male" else 1 for x in df["Sex"]]
    return df.iloc[:, 1:], df.iloc[:, 0]


if __name__ == "__main__":
    data = _prepare_data()
    run_bad_model(*data)
    run_model1(*data)
    run_model2(*data)
    run_model3(*data)

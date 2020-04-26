from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compress(X):
    X_fix = PCA(X.shape[1]//2).fit_transform(X)
    return TSNE(2).fit_transform(X_fix)

def plot(X, y_true, y_pred, outname):
    resolution = 500
    X2d_xmin, X2d_xmax = np.min(X[:,0]), np.max(X[:,0])
    X2d_ymin, X2d_ymax = np.min(X[:,1]), np.max(X[:,1])
    xx, yy = np.meshgrid(
        np.linspace(X2d_xmin, X2d_xmax, resolution),
        np.linspace(X2d_ymin, X2d_ymax, resolution))
    
    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X, y_pred)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    #plot
    plt.contourf(xx, yy, voronoiBackground, cmap="bwr")
    plt.scatter(X[:,0], X[:,1], c=y_true, s=8)
    plt.gray()
    plt.savefig(outname+".png")

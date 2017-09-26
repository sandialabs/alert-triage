# Code source: Gael Varoqueux
#              Andreas Mueller
# Modified for Documentation merge by Jaques Grobler
# License: BSD

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from alert_triage.active_learning import uncertainty

def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 200, 2
    np.random.seed(0)
    C = np.array([[1., 0.], [0., 1.]])
    X = np.r_[np.dot(np.random.randn(n, dim), C) + np.array([-2, 0]),
              np.dot(np.random.randn(n, dim), C) + np.array([2, 0])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

def main():
    h = .02  # step size in the mesh

    figure = pl.figure(figsize=(27, 9))
    X, y = dataset_fixed_cov()

    X = StandardScaler().fit_transform(X)
    select_size = int(0.025*len(y))
    train_indices = np.random.choice(np.arange(0, len(y)), select_size,
                                     replace=False)
    test_indices = [index for index in np.arange(0, len(y)) if index not in
                    train_indices.tolist()]

    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]
    class_0 = (y_train == 0)
    class_1 = (y_train == 1)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.96)

    x_min, x_max = X[:, 0].min() - .15, X[:, 0].max() + .15
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    """
    ax = pl.subplot(1, 1, 1)
    # Plot the training points
    class_0 = (y_train == 0)
    class_1 = (y_train == 1)
    ax.scatter(X_test[:,0], X_test[:,1], marker="o", color="gray", s=50)
    ax.scatter(X_train[class_0, 0], X_train[class_0, 1], c=y_train[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="H", s=450)
    ax.scatter(X_train[class_1, 0], X_train[class_1, 1], c=y_train[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=450)

    # and testing points
    #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    """
    ax = pl.subplot(1, 1, 1)
    clf = SVC(kernel="linear")
    active_learner = uncertainty.UncertaintySampling(budget=1, model=clf)
    active_learner.fit(X_train, y_train)

    for i in xrange(10):
        queried = active_learner.query(X, y, test_indices)
        train_indices = np.hstack([train_indices, queried])
        test_indices = [index for index in np.arange(0, len(y)) if index not in
                    train_indices.tolist()]
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]
        class_0 = (y_train == 0)
        class_1 = (y_train == 1)
        active_learner.fit(X_train, y_train)

    clf = active_learner.model
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.contour(xx, yy, Z, colors="black", linewidths=(0., 0., 0., 10., 0., 0.,
                                                      0.))

    # Plot also the training points
    ax.scatter(X_test[:, 0], X_test[:, 1], marker="o", color="gray", s=50)
    ax.scatter(X_train[class_0, 0], X_train[class_0, 1], c=y_train[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="H", s=450)
    ax.scatter(X_train[class_1, 0], X_train[class_1, 1], c=y_train[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=450)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_autoscale_on(True)

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()

if __name__ == "__main__":
    main()

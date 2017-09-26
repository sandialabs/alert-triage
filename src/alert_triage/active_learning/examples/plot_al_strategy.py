# Code source: Gael Varoqueux
#              Andreas Mueller
# Modified for Documentation merge by Jaques Grobler
# License: BSD

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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

    # Random Sampling
    ax = pl.subplot(1, 2, 1)
    select_size = 2
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

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for i in xrange(30):
        active_learner = sampling.RandomSampling()
        active_learner.budget = 1
        active_learner.data = X_test
        train_indices = np.hstack([train_indices,
                               active_learner.query(test_indices)])
        test_indices = [index for index in np.arange(0, len(y)) if index not in
                    train_indices.tolist()]

        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]
        class_0 = (y_train == 0)
        class_1 = (y_train == 1)

    # Plot also the training points
    ax.scatter(X_train[class_0, 0], X_train[class_0, 1], c=y_train[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="^", s=150)
    ax.scatter(X_train[class_1, 0], X_train[class_1, 1], c=y_train[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=150)
    ax.scatter(X_test[:, 0], X_test[:, 1], marker="o", color="gray")
    ax.set_xticks(())
    ax.set_yticks(())

    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = pl.subplot(1, 2, 2)
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


    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)

    for i in xrange(30):
        active_learner = sampling.UncertaintySampling()
        active_learner.model = clf
        active_learner.budget = 1
        active_learner.data = X_test
        train_indices = np.hstack([train_indices,
                               active_learner.query(test_indices)])
        test_indices = [index for index in np.arange(0, len(y)) if index not in
                    train_indices.tolist()]

        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]
        class_0 = (y_train == 0)
        class_1 = (y_train == 1)
        clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    Z = Z.reshape(xx.shape)
    #ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.contour(xx, yy, Z, colors="black", linewidths=(0., 0., 0., 2., 0., 0.,
                                                      0.))

    # Plot also the training points
    ax.scatter(X_train[class_0, 0], X_train[class_0, 1], c=y_train[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="^", s=150)
    ax.scatter(X_train[class_1, 0], X_train[class_1, 1], c=y_train[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=150)
    ax.scatter(X_test[:, 0], X_test[:, 1], marker="o", color="gray")

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #        size=15, horizontalalignment='right')

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()

if __name__ == "__main__":
    main()

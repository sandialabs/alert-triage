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

    x_min, x_max = X[:, 0].min() - .15, X[:, 0].max() + .15
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    class_0 = (y == 0)
    class_1 = (y == 1)

    # just plot the dataset first
    cm = pl.cm.RdBu
    """
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = pl.subplot(1, 2, 1)
    # Plot the training points
    ax.scatter(X[class_0, 0], X[class_0, 1], c=y[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="^", s=150)
    ax.scatter(X[class_1, 0], X[class_1, 1], c=y[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=150)
    # and testing points
    #ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    """
    ax = pl.subplot(1, 1, 1)
    clf = SVC(kernel="linear")
    clf.fit(X, y)

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
    ax.scatter(X[class_0, 0], X[class_0, 1], c=y[class_0],
               cmap=ListedColormap(['#FF0000', '#0000FF']),
               marker="H", s=450)
    ax.scatter(X[class_1, 0], X[class_1, 1], c=y[class_1],
               cmap=ListedColormap(['#0000FF', '#FF0000']),
               marker="s", s=450)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_autoscale_on(True)
    #ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #        size=15, horizontalalignment='right')

    #figure.subplots_adjust(left=.02, right=.98)
    pl.show()

if __name__ == "__main__":
    main()

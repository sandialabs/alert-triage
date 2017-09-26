import numpy


def logit(w):
    return 1.0/(1.0 + numpy.exp(-w))


def softmax(w, t=1.0):
    e = numpy.exp(w / t)
    dist = e / numpy.sum(e)
    return dist


def neg_log_likelihood(X, w, b=0.0, smooth_func=softmax):
    p_y_given_x = smooth_func(numpy.dot(X, w) + b)
    return -numpy.mean(numpy.log(p_y_given_x))


def hessian(func, X, w, delta=10e-5):
    """
    Calculate the Hessian matrix of func evaluated at params
    """
    dims = X.shape[1]
    hessian = numpy.zeros((dims, dims))
    tmpi = numpy.zeros(dims)
    tmpj = numpy.zeros(dims)

    for i in xrange(dims):
        tmpi[i] = delta
        params1 = X + tmpi
        params2 = X - tmpi

        for j in xrange(i, dims):
            tmpj[j] = delta
            deriv2 = (func(X=params2 + tmpj, w=w) -
                      func(X=params1 + tmpj, w=w))/delta
            deriv1 = (func(X=params2 - tmpj, w=w) -
                      func(X=params1 - tmpj, w=w))/delta
            hessian[i][j] = (deriv2 - deriv1)/delta

            # Since the Hessian is symmetric, spare me some calculations
            hessian[j][i] = hessian[i][j]
            tmpj[j] = 0
        tmpi[i] = 0
    return hessian


def compute_observed_information(X, w, b=0.0):
    return hessian(func=neg_log_likelihood, X=X, w=w)


def compute_expected_information(X, w, b=0.0):
    #f = features
    #F_ij = 2 f_i f_j sigma'(theta dot f)^2 / sigma(theta dot f)
    npar = X.shape[1]
    F = numpy.zeros([npar, npar])
    theta_dot_x = numpy.dot(X, w) + b
    F = 2 * numpy.dot(X.T, X) * numpy.var(theta_dot_x)**2
    F /= numpy.var(theta_dot_x)
    return numpy.mat(F).I


class AOptimalSampling(object):
    def __init__(self, weights=None, bias=0.0, budget=1):
        self.weights = weights
        self.bias = bias
        self.budget = budget

    def query(self, unlabeled_indices):
        if self.weights is None:
            self.weights = numpy.zeros(self.data.shape[1])
        labeled_indices

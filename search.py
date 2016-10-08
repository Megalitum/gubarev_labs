import numpy as np
from matplotlib import *
from matplotlib.pyplot import *

def recursive_lsq(w: np.array, y: np.array):
    assert (w.shape[0] == y.shape[0])
    m = w.shape[1]

    def theta_generator():
        a = 0
        beta = eta = np.dot(w[:, 0], w[:, 0])
        gamma = np.dot(w[:, 0], y)
        nu = gamma / beta
        theta = np.array([nu])
        vec = y - w[:, 0] * theta
        rss = np.dot(vec, vec)
        yield theta, rss
        H_inv = np.array([[1 / beta]])
        for i in range(1, m):
            h = np.dot(w[:, i], w[:, :i])
            a = np.dot(H_inv, h)
            eta = np.dot(w[:, i], w[:, i])
            gamma = np.dot(w[:, i], y)
            beta = eta - np.dot(h, a)
            if beta == 0:
                return
            nu = (gamma - np.dot(theta, h)) / beta
            theta -= nu * a
            theta = np.hstack((theta, nu))
            rss -= nu ** 2 * beta
            yield theta, rss
            h11 = H_inv + np.dot(a[:, np.newaxis], a[np.newaxis, :]) / beta
            h12 = -a[:, np.newaxis] / beta
            h21 = -a / beta
            h22 = 1 / beta
            H_inv = np.vstack((np.hstack((h11, h12)), np.hstack((h21, h22))))

    return theta_generator()


def coef_update(alpha: np.array, h, constaint):
    i = np.argmin(alpha)
    if (alpha[i] <= constaint):
        alpha[i] += h
        return True
    else:
        return False


class Search_Optimum(object):
    def __init__(self, Q, h, alpha_c, beta_c, x, y):
        self.Q = Q
        self.h = h
        self.alpha_c = alpha_c
        self.beta_c = beta_c
        self.x = x
        self.Y = y
        self.theta = None
        self.rss = np.inf
        self.changable = True  # change beta = true else alpha

    def coef_init(self):
        # n = self.y.size
        # xk = np.arange(0, n, 1)*self.delta don't use - replace self.x
        self.alpha_mask = np.zeros(self.Q)
        self.beta_mask = np.zeros(self.Q)
        self.alpha = np.ones(self.Q) * self.alpha_c[0]
        self.beta = np.ones(self.Q) * self.beta_c[0]
        self.alpha_best = self.alpha
        self.beta_best = self.beta
        self.y_appr = self.Y

    def coefs_update(self):
        if coef_update(self.alpha, self.h[0], self.alpha_c[1]) is True:
            return True
        elif coef_update(self.beta, self.h[1], self.beta_c[1]) is True:
            self.alpha = np.ones(self.Q) * self.alpha_c[0]
            return True
        else:
            return False

    def iterate(self):
        self.coef_init()
        while (True):
            W = self.coef_compile()

            # theta = None
            # for theta in recursive_lsq(W, self.Y):
            #     pass
            # if(theta[1]<self.rss):
            #     self.theta = theta[0]
            #     self.rss = theta[1]
            #     self.alpha_best = self.alpha.copy()
            #     self.beta_best = self.beta.copy()

            theta = np.linalg.lstsq(W, self.Y)
            y_appr = np.dot(W,theta[0])
            rss = np.linalg.norm(self.Y - y_appr)
            # print('-----------------------------')
            # print('alpha=',self.alpha, 'alpha_best=', self.alpha_best)
            # print('beta =', self.beta, 'beta_best =', self.beta_best)
            # print('current rss=', rss)
            # print('best rss   =',self.rss)
            # remember: assign variable in numpy as reference
            if (rss < self.rss):
                self.theta = theta[0]
                self.rss = rss
                self.alpha_best = self.alpha.copy()
                self.beta_best = self.beta.copy()
                self.y_appr = y_appr.copy()
            if self.coefs_update() is False:
                break

    def coef_compile(self):
        '''
        construct matrix M; Y = W*X
        Example Q = 2
            m[k] = [cos(delta* k *beta_1)*exp(-k*delta*alpha_1),
                    cos(delta* k *beta_2)*exp(-k*delta*alpha_2),
                    sin(delta* k *beta_1)*exp(-k*delta*alpha_1),
                    sin(delta* k *beta_2)*exp(-k*delta*alpha_2)
        :return:
        '''
        part_exp = np.exp(-self.x[:, np.newaxis] * self.alpha)
        part_beta = self.x[:, np.newaxis] * self.beta
        part_left = np.multiply(np.cos(part_beta), part_exp)
        part_right = np.multiply(np.sin(part_beta), part_exp)
        return np.hstack((part_left, part_right))

    def __str__(self):
        return 'theta=' + str(self.theta) + '\n' + 'rss=' + str(self.rss) + \
               '\nalpha_best=' + str(self.alpha_best) + '\nbeta_best=' + str(self.beta_best)

    def params(self):
        return np.concatenate((self.theta,self.alpha_best,self.beta_best)),\

    def y_appr(self):
        return self.y_appr

def test():
    N = 500
    Q = 2
    h = np.array([0.1, 0.1])
    # np.save('search_test_data/y_10.npy', y_real)
    npzfiles = np.load('lab1_data/x-y-500.npz')
    x = npzfiles['x'][:N]
    y = npzfiles['y'][:N]
    alpha_c = np.array([0.01, 5])
    beta_c = np.array([0.01, 8])
    # delta = 0.0800769873503

    search = Search_Optimum(Q, h, alpha_c, beta_c, x, y)
    search.iterate()
    params = search.params()
    y_appr = search.y_appr()
    plot_y(x, y, y_appr)

    #print(search)
    #print(search.params())
    #return search.params()

def plot_y(x, y, y_appr):
    figure(figsize=(15,10))
    #print('y_real:\n',y_real[:50])
    plot(x, y,'-r', label='real')
    plot(x, y_appr,'-b', label = 'approximation')
    legend(loc='upper right')
    show()

test()

import numpy as np
from  scipy.optimize import basinhopping, minimize, fmin_bfgs
from matplotlib import *
from matplotlib.pyplot import *

def plot_y(x, y, y_appr, rss=0.0):
    figure(figsize=(15, 10))
    # print('y_real:\n',y_real[:50])
    plot(x, y, '-r', label='real')
    plot(x, y_appr, '-b', label='approximation')
    annotate('rss = ' + str(rss), (1, 7), backgroundcolor='w')
    legend(loc='upper right')

    show()

def plot_lambda(x, y, x_appr, y_appr):
    figure(figsize=(15, 10))
    # print('y_real:\n',y_real[:50])
    plot(x, y, 'go', color = (1,0,0),label='real')
    plot(x_appr, y_appr, 'go', color=(0,0,1), label='approximation')
    legend(loc='upper right')

    show()

class Search_Min(object):
    def __init__(self, x, y, Q, fc=None, fs=None, alpha=None, beta=None,\
                 fc_c=None, fs_c=None, alpha_c=None, beta_c=None):
        """
        init object
        fc, fs, alpha, beta: initial params
        """
        self.x = x
        self.y = y
        self.Q = Q
        self.fc = fc
        self.fs = fs
        self.alpha = alpha
        self.beta = beta
        self.params = self.set_params() # params will be initial points in minimization
        assert(self.params.size == self.Q*4)
        self.alpha_c = alpha_c
        self.beta_c = beta_c
        self.fc_c = fc_c
        self.fs_c = fs_c
        self.N = y.shape[0]
        if fc_c is None:
            self.fc_c = np.array([None, None])
        if fs_c is None:
            self.fs_c = np.array([None, None])


    def set_params(self):
        """
        All params concatenate in one array
        :return: array
        """
        return np.concatenate((self.fc, self.fs, self.alpha, self.beta))

    def get_constaints(self):
        """
        concatenate in tuple constraint for all params:fc, fs, alpha and beta
        :return: tuple constraints
        """
        fc_c = np.tile(self.fc_c, (self.Q,1))
        fs_c = np.tile(self.fs_c, (self.Q,1))
        a_c = np.tile(self.alpha_c, (self.Q,1))
        b_c = np.tile(self.beta_c, (self.Q,1))
        #return tuple(map(tuple,np.vstack((fc_c, fs_c, a_c, b_c)).tolist()))
        return np.vstack((fc_c, fs_c, a_c, b_c)).tolist()


    #not used
    def y_x(self):
        def y_k_x(self, x):
            """
            evaluate y_k(or model) at point x
            :param x: ndarray
            :return: ndarray
            """
            shape = (self.N, 1)
            n_alpha = np.tile(self.alpha, shape)
            n_beta = np.tile(self.beta, shape)
            n_a_x = x[:, np.newaxis] * n_alpha
            n_b_x = x[:, np.newaxis] * n_beta
            return np.sum(self.fc * np.cos(n_b_x) * np.exp(-n_a_x) + self.fs * np.sin(n_b_x)\
                          * np.exp(-n_a_x), axis=1)

        def derivate_x(self,x):
            """
            derivate y_k(or model) at point x
            :param x: ndarray
            :return: ndarray
            """
            shape = (self.N, 1)
            n_alpha = np.tile(self.alpha, shape)
            n_beta = np.tile(self.beta, shape)
            n_a_x = x[:, np.newaxis] * n_alpha
            n_b_x = x[:, np.newaxis] * n_beta
            return np.sum(((-self.fc)*self.beta*np.sin(n_b_x)+\
                          self.fs*self.beta*np.cos(n_b_x)+
                          (-self.alpha)*self.fc*np.cos(n_b_x)+\
                          (-self.alpha)*self.fs*np.sin(n_b_x))*\
                          np.exp(-n_a_x), axis = 1)

        def fun_x_sum_squares(self,x):
            return (self.y - self.y_k(x))**2

        def derivate_x_sum_squares(self, x):
            return 2 * (self.y - self.y_k(x)) * self.derivate_x(x)

        def fun_dfun__x_sum_squares(self, x):
            """
            retunr f(x), df(x) (or derived f) in point x
            :param x: ndarray
            :return: ndarray
            """
            shape = (self.N, 1)
            n_alpha = np.tile(self.alpha, shape)
            n_beta = np.tile(self.beta, shape)
            n_a_x = x[:, np.newaxis] * n_alpha
            n_b_x = x[:, np.newaxis] * n_beta
            b_cos = np.cos(n_b_x)
            b_sin = np.sin(n_b_x)
            a_exp = np.exp(-n_a_x)
            y_k = np.sum(self.fc * b_cos * np.exp(-n_a_x) + self.fs * b_sin \
                         * np.exp(-n_a_x), axis=1)
            d_y_k = np.sum(((-self.fc) * self.beta * b_sin \
                            + self.fs * self.beta * b_cos \
                            + (-self.alpha) * (self.fc * b_cos + self.fs * b_sin)) \
                           * np.exp(-n_a_x), axis=1)
            return (self.y - y_k) ** 2, \
                   2 * (self.y - y_k) * d_y_k

        return y_k_x, derivate_x, fun_x_sum_squares,

    def y_params(self):
        """
        [0]: y; [1]:y'; [2]:y,y'
        :return:
        """
        def y_k_params(params):
            """
            eval y_k at params
            :param params: ndarray (size = 4*Q)
            :return: derived y_k at point params
            """
            [fc, fs, alpha, beta] = np.split(params, 4)
            assert(fc.size==fs.size==alpha.size==beta.size)
            shape = (self.N, 1)
            n_alpha = np.tile(alpha, shape)
            n_beta = np.tile(beta, shape)
            n_a_x = self.x[:, np.newaxis] * n_alpha
            n_b_x = self.x[:, np.newaxis] * n_beta

            return np.sum(fc * np.cos(n_b_x) * np.exp(-n_a_x) + fs * np.sin(n_b_x) \
                          * np.exp(-n_a_x), axis=1)

        def dy_k_params(params):
            """
            partitial derivation y by fc, fs, alpha, beta
            :param params: ndarray
            :return: ndarary (shape = (N, 4*Q))
            """
            [fc, fs, alpha, beta] = np.split(params, 4)
            m_ = np.multiply  # ([[1,2],[3,4]], [[1,2],[3,4]]) --> [[1,4],[9,16]]
            o_ = np.outer  # ([1,2,3], [1,2]) --> [[1,2],[2,4],[3,6]]
            b_x = np.outer(self.x, beta)
            exp_a_x = np.exp(-np.outer(self.x, alpha))
            p_fc = np.multiply(np.cos(b_x), exp_a_x)
            p_fs = np.multiply(np.sin(b_x), exp_a_x)
            p_alpha = np.outer(self.y_params()[0](params), -alpha)
            p_beta = m_(m_(-o_(self.x, fc), np.sin(b_x)) + m_(o_(self.x, fs), np.cos(b_x)), exp_a_x)
            return np.hstack((p_fc, p_fs, p_alpha, p_beta))

        return y_k_params, dy_k_params


    def sum_squered(self):

        def f_params_sum_squares(params):
            return np.sum((self.y-self.y_params()[0](params))**2)

        def df_params_sum_squares(params):
            y_k = self.y_params()[0](params)
            return np.sum(2*(y_k-self.y)[:,np.newaxis]*self.y_params()[1](params), axis = 0)

        def f_df_params_sum_squares(params):
            y_k = self.y_params()[0](params)
            return np.sum((self.y-y_k)**2), \
                   np.sum(2*(y_k-self.y)[:,np.newaxis]*self.y_params()[1](params), axis = 0)

        return f_params_sum_squares, df_params_sum_squares, f_df_params_sum_squares

###################### METHODS#########################
    def min_basinhopping(self):
        """
        Find the global minimum of a function using the basin-hopping algorithm
        :return: result = [x, f(x)]
        """
        self.bounds = self.get_constaints()
        print(self.bounds)
        print(self.params)
        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": self.bounds}

        def print_fun(x, f, accepted):
            print("at minimum %.4f accepted %d" % (f, int(accepted)))
        res = basinhopping(self.sum_squered()[2], self.params, niter=300, \
                           minimizer_kwargs=minimizer_kwargs, stepsize=0.05, callback=print_fun)

        print(res.x, res.fun)
        return res

    def min_neldermead(self):
        """
        Minimization of scalar function of one or more variables using the Nelder-Mead algorithm.
        :return:
        """
        def print_fun(x):
            print("at minimum {}",(self.y_params()[0](x)))
        res = minimize(self.sum_squered()[0], self.params, method='Nelder-Mead', callback=print_fun,
                       options={'maxiter': 10 ** 5, 'maxfev': 10 ** 5, 'disp': True},tol=10**-2)
        return res


    def min_fmin_bfgs(self):
        res = fmin_bfgs(self.sum_squered()[0], self.params,self.sum_squered()[1] ,\
                        disp = True, maxiter = 200)
        return res

    def min_bfgs(self):
        """
        Minimization of scalar function of one or more variables using the BFGS algorithm.
        :return:
        """
        jac = self.sum_squered()[1]
        def print_fun(x):
            print("at minimum {}",(self.y_params()[0](x)))
        res = minimize(self.sum_squered()[0], self.params, method='BFGS', callback=print_fun,
                       jac=jac,
                       options={'maxiter': 10 ** 5, 'maxfev': 10 ** 5, 'disp': True},tol=10**-2)
        return res


def test_yk_x_derivate():
    eigenvalues = np.load('lab1_data/points.npy')
    f_cc = np.load('lab1_data/f_c.npy')
    f_ss = np.load('lab1_data/f_s.npy')
    npzfiles = np.load('lab1_data/x-y-500.npz')
    x = npzfiles['x']
    y = npzfiles['y']
    alpha = -eigenvalues.real
    beta = eigenvalues.imag

    min = Search_Min(x, y, 30, f_cc, f_ss, alpha, beta)
    print(y[:4]==min.y_x()[0](x)[:4],y[:4],min.y_x()[0](x)[:4])
    #[ True  True False False] [-7.96501738 -4.38905442 -1.88638799 -0.62852533] [-7.96501738 -4.38905442 -1.88638799 -0.62852533]
    print(min.y_x()[1](x).shape, min.y_x()[1](x)[:5])
    #(500,) [ 48.92306936  38.87632396  23.29505891   8.75301637  -0.28295399]
#test_yk_derivate()

def test_methods():
    eigenvalues = np.load('lab1_data/points.npy')
    f_cc = np.load('lab1_data/f_c.npy')
    f_ss = np.load('lab1_data/f_s.npy')
    npzfiles = np.load('lab1_data/x-y-500.npz')
    x = npzfiles['x']
    y = npzfiles['y']
    alpha = -eigenvalues.real
    beta = eigenvalues.imag

    Q=10
    np.random.seed(0)
    alpha = np.random.random(Q)*7
    beta = np.random.random(Q)*10

    min = Search_Min(x, y, Q, f_cc[:Q], f_ss[:Q], alpha, beta,\
                     #fc_c=np.array([-1.2,1.2]), fs_c=np.array([-1.2,1.2]),\
                     alpha_c=np.array([0,7]), beta_c=np.array([0,10]))
    res = min.min_basinhopping()
    y_appr = min.y_params()[0](res.x)
    lamb = res.x

    #res = min.min_neldermead()
    #y_appr = min.y_params()[0](res.x)

    #res= min.min_fmin_bfgs()
    #y_appr = min.y_params()[0](res)

    #res = min.min_bfgs()
    #y_appr = min.y_params()[0](res.x)

    plot_y(np.arange(0, y.size, 1), y, y_appr)
    plot_lambda(-eigenvalues[eigenvalues.imag>=0].real, eigenvalues[eigenvalues.imag>=0].imag, lamb[2*Q:3*Q], lamb[3*Q:])
test_methods()








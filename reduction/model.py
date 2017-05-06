from pylab import *
from collections import Counter


def generate_params(count_real, count_imaginary, count_mixed, bounds_real, bounds_imaginary, f_generator):
        assert (count_imaginary % 2 == 0) and (count_mixed % 2 == 0), "Complex eigenvalues must be paired."
        eig_real = np.random.uniform(bounds_real[0], bounds_real[1], count_real)
        eig_imaginary = np.random.uniform(bounds_imaginary[0], bounds_imaginary[1], count_imaginary // 2) * 1j
        mean_mixed = [np.mean(bounds_real), np.mean(bounds_imaginary)]
        scale_mixed = [np.std(bounds_real) / 2, np.std(bounds_imaginary) / 2]
        eig_mixed = np.random.normal(mean_mixed, scale_mixed, size=(count_mixed // 2, 2))
        eig_mixed[:, 0] = np.clip(eig_mixed[:, 0], *bounds_real)
        eig_mixed[:, 1] = np.clip(eig_mixed[:, 1], *bounds_imaginary)
        eig_mixed = eig_mixed[:,0] + eig_mixed[:,1] * 1j
        f_vals = np.concatenate((f_generator(count_real), f_generator(count_imaginary // 2),
                                 f_generator(count_mixed // 2)), axis=1)
        return np.concatenate((eig_real, eig_imaginary, eig_mixed)), f_vals


class ObservableSystem(object):
    def __init__(self, **kwargs):
        eigenvalues = kwargs.get('eigenvalues', None)
        f_params = kwargs.get('f_params', None)
        delta = kwargs.get('delta', None)
        load_path = kwargs.get('path', None)
        if eigenvalues is None and load_path is None:
            self.eigenvalues, (self.f_cc, self.f_ss) = generate_params(4, 6, 20, (-8, 0), (0, 8), self.generate_f)
        elif load_path is not None:
            self.eigenvalues = load(load_path + '/points.npy')
            self.f_cc = load(load_path + '/f_c.npy')
            self.f_ss = load(load_path + '/f_s.npy')
        else:
            self.eigenvalues = eigenvalues
            if f_params is None:
                self.f_cc, self.f_ss = self.generate_f(len(eigenvalues))
            else:
                self.f_cc, self.f_ss = f_params
        sorted_eig = np.argsort(self.eigenvalues.imag)
        self.eigenvalues = self.eigenvalues[sorted_eig]
        self.f_cc = self.f_cc[sorted_eig]
        self.f_ss = self.f_ss[sorted_eig]
        self.count_real = np.sum(np.isclose(self.eigenvalues.imag, 0))
        print('Real eigenvalues: : ', self.count_real)
        if delta is None:
            self.delta = self.calculate_delta()
        else:
            self.delta = delta
        print('Eigenvalues: ', self.eigenvalues)
        print('f_c: ', self.f_cc)
        print('f_s: ', self.f_ss)
        print('Delta: ', self.delta)

    @staticmethod
    def generate_f(count):
        """
        Generates random numbers of given size uniformly in (-1, -0.9) join (0.9, 1).
        """
        arr = np.random.uniform(0.9, 1.1, (2, count))
        arr[arr > 1] -= 2
        return arr

    @staticmethod
    def response(x, params, delta=0.05):
        """
        Creates impulse reaction with given parameters and returns it.
        Assumes it's argument has `params = (f_c, f_s, alpha, beta)` structure.
        f_c = params[:q]
        f_s = params[q:2*q]
        alpha = params[2*q:3*q]
        beta = params[3*q:]
        """
        q = params.shape[0] // 4
        a_d = delta * params[2 * q:3 * q]
        b_d = delta * params[3 * q:]
        x_b = x[:, newaxis] * b_d
        coef = params[:q] * np.cos(x_b) + params[q:2 * q] * np.sin(x_b)
        exp_val = np.exp(- x[:, newaxis] * a_d)
        return real(sum(multiply(coef, exp_val), axis=1))

    @staticmethod
    def unit_response(x, params, delta=0.05, jac=True):
        """
        Creates impulse reaction with given parameters and returns it.
        Assumes it's argument has `params = [f_c, f_s, alpha, beta]` structure.
        f_c = params[0]
        f_s = params[1]
        alpha = params[2]
        beta = params[3]
        """
        x_scaled = x * delta
        x_b = x_scaled * params[3]
        exp_val = np.exp(- x_scaled * params[2])
        exp_coef = params[0] * np.cos(x_b) + params[1] * np.sin(x_b)
        if not jac:
            return exp_coef * exp_val
        dfc = exp_val * cos(x_b)
        dfs = exp_val * sin(x_b)
        da = - exp_coef * x_scaled
        db = exp_val * (params[1] * np.cos(x_b) - params[0] * np.sin(x_b)) * x_scaled
        return real(exp_coef * exp_val), real(vstack((dfc, dfs, da, db)))

    def calculate_delta(self):
        return min(2 / (5 * max(abs(self.eigenvalues.real))), np.pi / (5 * max(abs(self.eigenvalues.imag))))

    @property
    def all_eigenvalues(self):
        complex_eigenvalues = np.tile(self.eigenvalues[self.count_real:], (2, 1))
        complex_eigenvalues[1].imag *= -1
        return np.concatenate((self.eigenvalues[:self.count_real],*complex_eigenvalues))

    def __call__(self, x):
        params = concatenate((self.f_cc, self.f_ss, -self.eigenvalues.real, self.eigenvalues.imag))
        return self.response(x, params, self.delta)


class SimplifiedObservableSystem(ObservableSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def generate_f(count):
        """
        Generates random numbers of given size uniformly in (-1, -0.9) join (0.9, 1).
        """
        return np.ones((2, count))
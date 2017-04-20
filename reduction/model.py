from pylab import *
from collections import Counter


def generate_params(count_real, count_imaginary, count_mixed, bounds_real, bounds_imaginary, f_generator):
        assert (count_imaginary % 2 == 0) and (count_mixed % 2 == 0), "Complex eigenvalues must be paired."
        eig_real = random.uniform(bounds_real[0], bounds_real[1], count_real)
        eig_imaginary = random.uniform(bounds_imaginary[0], bounds_imaginary[1], count_imaginary // 2) * 1j
        eig_imaginary = np.tile(eig_imaginary, 2)
        mean_mixed = [np.mean(bounds_real), np.mean(bounds_imaginary)]
        scale_mixed = [np.std(bounds_real), np.std(bounds_imaginary)]
        eig_mixed = random.normal(mean_mixed, scale_mixed, count_mixed // 2)
        eig_mixed = np.tile(eig_mixed, 2)
        f_vals = np.concatenate((f_generator(count_real), np.tile(f_generator(count_imaginary // 2), 2),
                                 np.tile(f_generator(count_mixed // 2), 2)), axis=1)
        return np.concatenate((eig_real, eig_imaginary, eig_mixed)), f_vals


class ObservableSystem(object):
    def __init__(self, eigenvalues=None, f_params = None):
        if eigenvalues is None:
            self.eigenvalues, (self.f_cc, self.f_ss) = generate_params(4, 6, 20, (-8, 8), ())
            self.eigenvalues = load('lab1_data/points.npy')
            if f_params is None:
                self.f_cc = load('lab1_data/f_c.npy')
                self.f_ss = load('lab1_data/f_s.npy')
            else:
                self.f_cc = f_params[0]
                self.f_ss = f_params[1]
        else:
            # TODO: Fix f generation bug (mult by taylor).
            cnt = Counter(eigenvalues)
            unique_eigenvalues = cnt.keys()
            eigen_count = len(unique_eigenvalues)
            f_c, f_s = self.generate_f((2, eigen_count))
            self.eigenvalues = concatenate(tuple(repeat(value, cnt[value]) for value in unique_eigenvalues))
            self.f_cc = concatenate(tuple(repeat(f_c[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
            self.f_ss = concatenate(tuple(repeat(f_s[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
        self.delta = self.calculate_delta()
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

    def __call__(self, x):
        params = concatenate((self.f_cc, self.f_ss, -self.eigenvalues.real, self.eigenvalues.imag))
        return self.response(x, params, self.delta)


class SimplifiedObservableSystem(ObservableSystem):
    def __init__(self, eigenvalues=None):
        if eigenvalues is None:
            eigenvalues = load('lab1_data/points.npy')
        super().__init__(eigenvalues)

    @staticmethod
    def generate_f(count):
        """
        Generates random numbers of given size uniformly in (-1, -0.9) join (0.9, 1).
        """
        return np.ones((2, count))
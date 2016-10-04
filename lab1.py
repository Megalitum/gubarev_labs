from pylab import *
import scipy as sp
import seaborn as sns

from collections import Counter

def generate_f(size):
    """
    Generates random numbers of given size uniformly in (-1, -0.9) join (0.9, 1).
    """
    arr = random.uniform(0.9, 1.1, size)
    arr[arr > 1] -= 2
    return arr

def get_impulse_sum_optimized(f_c, f_s, alpha, beta, delta=0.05):
    """
    Creates sum calculator with given parameters and returns it.
    """
    a_d = delta * alpha
    b_d = delta * beta
    def sum_calculator(x):
        x_b = x[:, newaxis] * b_d
        coef = f_c * np.cos(x_b) + f_s * np.sin(x_b)
        exp_val = np.exp(- x[:, newaxis] * a_d)
        return sum(multiply(coef, exp_val), axis=1)
    return sum_calculator

def calculate_delta(eigenvalues):
    return min(2 / (5 * max(abs(eigenvalues.real))), np.pi / (5 * max(abs(eigenvalues.real))))


class InitialImpulse(object):
    def __init__(self, eigenvalues=None):
        if eigenvalues is None:
            self.eigenvalues = load('lab1_data/points.npy')
            self.f_cc = load('lab1_data/f_c.npy')
            self.f_ss = load('lab1_data/f_s.npy')
        else:
            cnt = Counter(eigenvalues)
            unique_eigenvalues = cnt.keys()
            eigen_count = len(unique_eigenvalues)
            f_c, f_s = generate_f((2, eigen_count))
            self.eigenvalues = concatenate(tuple(repeat(value, cnt[value]) for value in unique_eigenvalues))
            self.f_cc = concatenate(tuple(repeat(f_c[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
            self.f_ss = concatenate(tuple(repeat(f_s[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
        self.delta = calculate_delta(self.eigenvalues)
        print('Eigenvalues: ', self.eigenvalues)
        print('f_c: ', self.f_cc)
        print('f_s: ', self.f_ss)
        print('Delta: ', self.delta)

    def functor(self):
        return get_impulse_sum_optimized(self.f_cc, self.f_ss, -self.eigenvalues.real, self.eigenvalues.imag, self.delta)

impulse = InitialImpulse()

def generate_cost_func(support, complexity):
    """
    Example approximator.
    """
    value = impulse.functor()(support)
    delta = impulse.delta
    def approximator(vec):
        params = vec.reshape((4, complexity))
        return mean(abs(value -  get_impulse_sum_optimized(*params, delta)(support)))
    return approximator

Q = 6
appr = generate_cost_func(arange(0, 500, 1), Q)
f_cc = impulse.f_cc[impulse.eigenvalues.real == 0]
f_ss = impulse.f_ss[impulse.eigenvalues.real == 0]
alpha = zeros(Q)
beta = impulse.eigenvalues[impulse.eigenvalues.real == 0].imag
init_params = concatenate((f_cc, f_ss, alpha, beta))
#print(appr(init_params))
solution = sp.optimize.fmin_bfgs(appr, init_params)
#print(solution.reshape((4, Q)))
#solution = init_params
y_appr = get_impulse_sum_optimized(*solution.reshape((4, Q)), impulse.delta)

figure(figsize=(15,10))
X = arange(0, 500, 1)
plot(X, impulse.functor()(X))
plot(X, y_appr(X))
show()
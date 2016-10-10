from pylab import *
from scipy import optimize
import seaborn as sns

from collections import Counter
import itertools


def generate_f(size):
    """
    Generates random numbers of given size uniformly in (-1, -0.9) join (0.9, 1).
    """
    arr = random.uniform(0.9, 1.1, size)
    arr[arr > 1] -= 2
    return arr


def impulse_vec(x, params, delta=0.05):
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
    coef = params[:q] * np.cos(x_b) + params[q:2*q] * np.sin(x_b)
    exp_val = np.exp(- x[:, newaxis] * a_d)
    return sum(multiply(coef, exp_val), axis=1)


def impulse_with_gradient(x, params, delta=0.05):
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
    dfc = exp_val * cos(x_b)
    dfs = exp_val * sin(x_b)
    exp_coef = params[0] * np.cos(x_b) + params[1] * np.sin(x_b)
    da = - exp_coef * x_scaled
    db = exp_val * (params[1] * np.cos(x_b) - params[0] * np.sin(x_b)) * x_scaled
    return exp_coef * exp_val, vstack((dfc, dfs, da, db))

def impulse(x, params, delta=0.05):
    """
    Creates impulse reaction with given parameters and returns it.
    Assumes it's argument has `params = [f_c, f_s, alpha, beta]` structure.
    f_c = params[0]
    f_s = params[1]
    alpha = params[2]
    beta = params[3]
    """
    a_d = delta * params[2]
    x_b = x * delta * params[3]
    coef = params[0] * np.cos(x_b) + params[1] * np.sin(x_b)
    exp_val = np.exp(- x * a_d)
    return sum(multiply(coef, exp_val), axis=1)


class ObservableSystem(object):
    def __init__(self, eigenvalues=None):
        if eigenvalues is None:
            self.eigenvalues = load('lab1_data/points.npy')
            self.f_cc = load('lab1_data/f_c.npy')
            self.f_ss = load('lab1_data/f_s.npy')
        else:
            # TODO: Fix f generation bug (mult by taylor).
            cnt = Counter(eigenvalues)
            unique_eigenvalues = cnt.keys()
            eigen_count = len(unique_eigenvalues)
            f_c, f_s = generate_f((2, eigen_count))
            self.eigenvalues = concatenate(tuple(repeat(value, cnt[value]) for value in unique_eigenvalues))
            self.f_cc = concatenate(tuple(repeat(f_c[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
            self.f_ss = concatenate(tuple(repeat(f_s[i], cnt[value]) for i, value in enumerate(unique_eigenvalues)))
        self.delta = self.calculate_delta()
        print('Eigenvalues: ', self.eigenvalues)
        print('f_c: ', self.f_cc)
        print('f_s: ', self.f_ss)
        print('Delta: ', self.delta)

    def calculate_delta(self):
        return min(2 / (5 * max(abs(self.eigenvalues.real))), np.pi / (5 * max(abs(self.eigenvalues.imag))))

    def __call__(self, x):
        params = concatenate((self.f_cc, self.f_ss, -self.eigenvalues.real, self.eigenvalues.imag))
        return impulse_vec(x, params, self.delta)

observable = ObservableSystem()


def generate_cost_func(domain, noisy, type='l2', delta=0.05):
    """
    Cost function builder.
    Assumes vec has (4,) shape.
    vec[0] = f_c
    vec[1] = f_s
    vec[2] = alpha
    vec[3] = beta
    """
    if type == 'l_inf':
        def cost_func(vec):
            approx = impulse(domain, vec, delta)
            return max(abs(noisy - approx)) / domain.shape[0]
        return cost_func
    elif type == 'l2':
        def cost_func(vec):
            approx, gradient = impulse_with_gradient(domain, vec, delta)
            return sum((noisy - approx)**2), 2 * ((approx - noisy) * gradient).sum(axis=1)
        return cost_func
    else:
        raise NotImplemented()

def starting_points():
    yield from itertools.product([1], [1], arange(0,7,1), arange(-5,6,1))


def perform_approximaiton(domain, max_q, type='l2'):
    minimizer_kwargs = {"method":"L-BFGS-B", "jac":type == 'l2',
                        "bounds":((None, None), (None, None), (0, 6), (-10, 10))}
    noisy = observable(domain)
    for q in range(max_q):
        cost_func = generate_cost_func(domain, noisy, type)
        best_sol = None
        for start_point in starting_points():
            print('>', best_sol.fun if best_sol is not None else None)
            sol = optimize.basinhopping(cost_func, start_point, minimizer_kwargs=minimizer_kwargs)
            if best_sol is None or sol.fun < best_sol.fun:
                best_sol = sol
        noisy -= impulse_vec(domain, best_sol.x, observable.delta)
        print("Status: ", best_sol.message)
        print("f, f':", cost_func(best_sol.x))
        yield best_sol

max_q = 8
N = 500
domain = arange(1, N + 1, 1)
result = None
type = 'l2'
for q, solution in enumerate(perform_approximaiton(domain, max_q, type)):
    if result is None:
        result = solution.x[:, newaxis]
    else:
        result = np.append(result, solution.x[:, newaxis], axis=1)
    y_real = observable(domain)
    y_appr = impulse_vec(domain, result.ravel(), observable.delta)
    plot(domain, y_real, '-r', label='real')
    plot(domain, y_appr, '-b', label='approximation')
    legend(loc='upper right')
    title(str(q + 1))
    show()

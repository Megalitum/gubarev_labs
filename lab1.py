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


def impulse(x, params, delta=0.05):
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


class ObservableSystem(object):
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
        self.delta = self.calculate_delta()
        print('Eigenvalues: ', self.eigenvalues)
        print('f_c: ', self.f_cc)
        print('f_s: ', self.f_ss)
        print('Delta: ', self.delta)

    def calculate_delta(self):
        return min(2 / (5 * max(abs(self.eigenvalues.real))), np.pi / (5 * max(abs(self.eigenvalues.imag))))

    def __call__(self, x):
        params = concatenate((self.f_cc, self.f_ss, -self.eigenvalues.real, self.eigenvalues.imag))
        return impulse(x, params, self.delta)

observable = ObservableSystem()


def generate_cost_func(domain, complexity, type='l2'):
    """
    Cost function builder.
    """
    noisy = observable(domain)
    delta = observable.delta
    if type == 'l_inf':
        def cost_func(vec):
            approx = impulse(domain, vec, delta)
            return max(abs(noisy - approx))
        return cost_func
    elif type == 'l2':
        def cost_func(vec):
            approx = impulse(domain, vec, delta)
            return mean((noisy - approx)**2)
        return cost_func

def starting_points(Q):
    # yield concatenate(([1,1,-1,1,1,-1,1,-1,-1,1,1,-1], zeros(Q*2)))
    # return
    f_gen = itertools.product(*itertools.tee([-1, 1], Q*2))
    for f_vec in f_gen:
        yield concatenate((f_vec, zeros(Q*2)))

Q = 6
N = 500
domain = arange(0, N, 1)
cost_func = generate_cost_func(domain, Q)
solution = None
st_point = None
for start_point in starting_points(Q):
    print('bump')
    sol = optimize.minimize(cost_func, start_point, method='BFGS',
                                 options={'maxiter': 10000})
    if solution is None or solution.fun > sol.fun:
        solution = sol
        st_point = start_point

print('Found solution:', solution.x)
print('With f=', solution.fun)
print('Starting from: ', st_point)
y_appr = impulse(domain, solution.x, observable.delta)

figure(figsize=(16,9))
y_real = observable(domain)
plot(domain, y_real,'-r', label='real')
plot(domain, y_appr,'-b', label = 'approximation')
legend(loc='upper right')
show()
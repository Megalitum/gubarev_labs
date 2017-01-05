import numpy as np
import scipy
import timeit
from scipy.linalg import toeplitz
import scipy.special
import itertools
from sklearn.metrics import r2_score
import warnings
from matplotlib import pyplot as plt

# <editor-fold desc="Global parameters">
N_eig = 15
eigenvalues = np.load('lab2_data/new_eigenvalues.npy')
fc = np.ones(N_eig)
fs = np.ones(N_eig)

with np.load("lab2_data/coef.npz") as storage:
    phi = storage['arr_0']
    u_coef = storage['arr_1']
    w = storage['arr_2']


# </editor-fold>

# <editor-fold desc="Errors">
def norm_error(eps, n):
    error = []
    for i in range(n):
        a = np.random.normal(scale=eps / 3)
        while abs(a) > eps:
            a = np.random.normal(scale=eps / 3)
        error.append(a)
    return np.array(error)


def uniform_error(eps, n):
    return np.random.uniform(-eps, eps, size=n)


def two_peaks_error(eps, n):
    error = []
    for i in range(int(n / 2)):
        a = np.random.normal(loc=-0.2, scale=eps / 3)
        while abs(a) > eps:
            a = np.random.normal(loc=-0.2, scale=eps / 3)
        error.append(a)
        a = np.random.normal(loc=0.2, scale=eps / 3)
        while abs(a) > eps:
            a = np.random.normal(loc=0.2, scale=eps / 3)
        error.append(a)
    if len(error) < n:
        a = np.random.normal(loc=0.2, scale=eps / 3)
        while abs(a) > eps:
            a = np.random.normal(loc=0.2, scale=eps / 3)
        error.append(a)
    return np.array(error)


# </editor-fold>

def calculate_a_b(delta):
    global eigenvalues
    alpha = [np.exp(el.real * delta) * np.cos(el.imag * delta) for el in eigenvalues]
    beta = [np.exp(el.real * delta) * np.sin(el.imag * delta) for el in eigenvalues]
    return np.array(alpha), np.array(beta)


def calculate_match_percent(h_real, h_low, h_high):
    return sum((h_real >= h_low) * (h_real <= h_high)) / len(h_real) * 100


def generate_pure_h(T, delta=0.1):
    N = int(T / delta)
    alpha, beta = calculate_a_b(delta)
    rho = np.sqrt(alpha ** 2 + beta ** 2)
    omega = np.arccos(alpha / rho)
    h = [np.sum(rho ** i * (fc * np.cos(omega * i) + fs * np.sin(omega * i))) for i in range(N)]
    return np.array(h), np.arange(N) * delta


def run_m1(*args, **kwargs):
    T = kwargs['T']
    eps = kwargs['eps']
    delta = kwargs['delta']
    error_generator = kwargs['err_gen']
    u0, u1 = 1, 0.7

    def gen_real_y_u_h(N, alpha, beta, u_init):
        u0, u1 = u_init
        rho = [np.sqrt(alpha[i] ** 2 + beta[i] ** 2) for i in range(len(alpha))]
        omega = [np.arccos(alpha[i] / rho[i]) for i in range(len(alpha))]
        h = [sum(
            [rho[j] ** i * (fc[j] * np.cos(omega[j] * i) + fs[j] * np.sin(omega[j] * i)) for j in range(len(omega))])
             for i in range(N)]
        u = [u0, u1] + [u1 ** i / u0 ** (i - 1) for i in range(2, N)]
        y = [sum([h[i - j] * u[j] for j in range(i + 1)]) for i in range(N)]
        return np.array(y), np.array(u), np.array(h)

    def generate_values(eps, delta, T, gen_error, u_init):
        """
        return: alpha, beta, y_real, u, h_real, error, y
        """
        np.random.seed(0)
        N = int(T / delta)
        alpha, beta = calculate_a_b(delta)
        y_real, u, h_real = gen_real_y_u_h(N, alpha, beta, u_init)
        error = gen_error(eps, N)
        y = y_real + error
        return alpha, beta, y_real, u, h_real, error, y

    def calc_h_central(u, y):
        h = [y[0] / u[0]]
        for i in range(1, len(y)):
            h.append((y[i] - sum([h[j] * u[i - j] for j in range(i)])) / u[0])
        return np.array(h)

    def calc_confidence_intervals(h_cent, eps, u0, u1):
        calc_lower = [h_cent[0] - eps / u0]
        calc_upper = [h_cent[0] + eps / u0]
        calc_lower += [h_cent[i] - eps * (1 + u1 / u0) / u0 for i in range(1, len(h_cent))]
        calc_upper += [h_cent[i] + eps * (1 + u1 / u0) / u0 for i in range(1, len(h_cent))]
        return np.array(calc_lower), np.array(calc_upper)

    alpha, beta, y_real, u, h_real, error, y = generate_values(eps, delta, T, error_generator, (u0, u1))
    h_cent = calc_h_central(u, y)
    h_low, h_up = calc_confidence_intervals(h_cent, eps, u0, u1)

    def measured_task():
        calc_confidence_intervals(calc_h_central(u, y), eps, u0, u1)
        return
    return print_stats(1, T, error_generator, eps, delta, h_low, h_up, measured_task, 20)


def run_m2(*args, **kwargs):
    T = kwargs['T']
    eps = kwargs['eps']
    delta = kwargs['delta']
    error_generator = kwargs['err_gen']

    def gen_real_y_u_h(delta, N, alpha, beta):
        global phi, u_coef, w
        rho = [np.sqrt(alpha[i] ** 2 + beta[i] ** 2) for i in range(len(alpha))]
        omega = [np.arccos(alpha[i] / rho[i]) for i in range(len(alpha))]
        # print(len(rho), len(omega))
        h = [sum(
            [rho[j] ** i * (fc[j] * np.cos(omega[j] * i) + fs[j] * np.sin(omega[j] * i)) for j in range(len(omega))])
             for i in range(N)]
        u = [sum([u_coef[i] * np.sin(w[i] * delta * k + phi[i]) for i in range(32)]) for k in range(N)]
        u /= max(map(abs, u))
        y = [sum([h[i - j] * u[j] for j in range(i + 1)]) for i in range(N)]
        return np.array(y), np.array(u), np.array(h)

    def generate_values(eps, delta, T, gen_error, if_plot=False, folder=''):
        """
        return: alpha, beta, y_real, u, h_real, error, y
        """
        np.random.seed(0)
        N = int(T / delta)
        K = N + 5
        alpha, beta = calculate_a_b(delta)
        y_real, u, h_real = gen_real_y_u_h(delta, N + K, alpha, beta)
        error = gen_error(eps, N + K)
        assert (len(error) == len(y_real))
        y = y_real + error
        return alpha, beta, y_real, u, h_real, error, y

    def random_combination(k, r, count):
        arr = np.arange(k)
        for i in range(count):
            yield np.random.permutation(arr)[:r]

    def calc_confidence_intervals(u, y, eps):
        # print(len(u))
        N = int((len(u) - 5) / 2)
        K = N + 5
        # The Toeplitz matrix has constant diagonals, with c as its first column and r as its first row.
        matr = toeplitz(c=u[N:N + K], r=u[N::-1])
        h_low = np.ones(N + 1) * -1000
        h_up = np.ones(N + 1) * 1000
        satisfied = 0
        y_sliced = y[N:].copy()
        if scipy.special.binom(K, N + 1) > 50000:
            subset_gen = random_combination(K, N + 1, 50000)
            warnings.warn('Random picking due to high number of combinations.', RuntimeWarning)
        else:
            subset_gen = itertools.combinations(range(K), N + 1)
        for subset in subset_gen:
            U = matr[list(subset)]
            y_vec = y_sliced[list(subset)]
            if np.linalg.cond(U, p=2) < 15 * len(U):  # condition number
                satisfied += 1
                U_inv = scipy.linalg.inv(U)
                h_cent_temp = np.dot(U_inv, y_vec)
                h_delta = eps * np.sum(abs(U_inv), axis=1)
                h_low = np.maximum(h_low, h_cent_temp - h_delta)
                h_up = np.minimum(h_up, h_cent_temp + h_delta)
        h_cent = (h_low + h_up) / 2
        return h_cent, h_low, h_up

    alpha, beta, y_real, u, h_real, error, y = generate_values(eps, delta, T, error_generator)
    h_cent, h_low, h_up = calc_confidence_intervals(u, y, eps)
    def measured_task():
        calc_confidence_intervals(u, y, eps)
        return
    return print_stats(2, T, error_generator, eps, delta, h_low, h_up, measured_task, 1)


def run_m3(*args, **kwargs):
    T = kwargs['T']
    eps = kwargs['eps']
    split_depth = kwargs['depth']
    no_stages = kwargs['J']
    error_generator = kwargs['err_gen']

    def calculate_response(delta, N, J, alpha, beta):
        length = 2 * N * J
        rho = np.sqrt(alpha ** 2 + beta ** 2)
        omega = np.arccos(alpha / rho)
        h = [np.sum(rho ** i * (fc * np.cos(omega * i) + fs * np.sin(omega * i))) for i in range(length)]
        u = np.hstack((np.ones(N), np.zeros(N)) * J)
        y = np.convolve(h, u)[:length]
        return np.array(y), np.array(u), np.array(h)

    def calculate_parameters(eps, split_depth, T, J, gen_error):
        """
        Returns alpha, beta, y_real, u, h_real, error, y_noisy.
        """
        delta = T / 2 ** (split_depth + 1)
        N = round(T / delta)
        length = 2 * N * J
        alpha, beta = calculate_a_b(delta)
        y_real, u, h_real = calculate_response(delta, N, J, alpha, beta)

        np.random.seed(0)
        error = gen_error(eps, length)

        y_noisy = y_real + error
        epoch_size = y_noisy.shape[0] // J
        y_noisy = y_noisy.reshape(J, epoch_size) # y_noisy[epoch, index]

        return dict(alpha=alpha, beta=beta, y=y_real, signal=u,
                    h=h_real, noise=error, y_noisy=y_noisy, delta=delta, error_gen=gen_error.__name__)

    def solve_epoch(y_epoch, detalization, h_prev=None):
        shift = split_depth - detalization
        sl = slice(2 ** shift - 1, None, 2 ** shift)
        y_s = y_epoch[sl]
        h_low, h_high = -np.inf, np.inf
        # upper triangle
        bound = 2 ** (detalization + 1)
        h = y_s[:bound].copy()
        h[1:] = y_s[1:bound] - y_s[:bound - 1]
        h_low_, h_high_ = h - 2 * eps, h + 2 * eps
        h_low_[0] += eps
        h_high_[0] -= eps
        h_low, h_high = np.maximum(h_low, h_low_), np.minimum(h_high, h_high_)
        if h_prev:
            h_low_[1::2], h_high_[1::2] = h_prev[0] - h_high_[0::2], h_prev[1] - h_low_[0::2]
            h_low, h_high = np.maximum(h_low, h_low_), np.minimum(h_high, h_high_)
        # lower triangle
        h = y_s[bound - 1:].copy()
        h[:-1] = y_s[bound - 1:2 * bound - 2] - y_s[bound:]
        h_low_, h_high_ = h - 2 * eps, h + 2 * eps
        h_low_[-1] += eps
        h_high_[-1] -= eps
        h_low, h_high = np.maximum(h_low, h_low_), np.minimum(h_high, h_high_)
        if h_prev:
            h_low_[1::2], h_high_[1::2] = h_prev[0] - h_high_[0::2], h_prev[1] - h_low_[0::2]
            h_low, h_high = np.maximum(h_low, h_low_), np.minimum(h_high, h_high_)
        return h_low, h_high

    def solve_detalization(y_noisy, detalization, h_prev=None):
        h_min = []
        h_max = []
        for y_epoch in y_noisy:
            h_min_max = solve_epoch(y_epoch[:-1], detalization, h_prev)
            h_min.append(h_min_max[0])
            h_max.append(h_min_max[1])
        return np.amax(h_min, axis=0), np.amin(h_max, axis=0)

    def solve_tixonov(y_noisy, split_depth):
        h_current = None
        for detal in range(split_depth + 1):
            h_current = solve_detalization(y_noisy, detal, h_current)
        return h_current

    params = calculate_parameters(eps, split_depth, T, no_stages, error_generator)
    h_low, h_up = solve_tixonov(params['y_noisy'], split_depth)
    def measured_task():
        solve_tixonov(params['y_noisy'], split_depth)
        return
    return print_stats(3, T, error_generator, eps, params['delta'], h_low, h_up, measured_task, 20)


def print_stats(method, T, err_gen, eps, delta, h_low, h_up, task, times):
    """
    Returns delta, cover_percent, match_percent, r2, stdev, time.  
    """
    err_name = err_gen.__name__
    h_real, series_real = generate_pure_h(T, 0.01)
    series = np.arange(0, delta * len(h_low), delta)
    assert(len(series) == len(h_low))
    match_percent, cover_percent, r2, stdev, *other = test_h_prediction(series_real, h_real, series, h_low, h_up)
    time = timeit.timeit('measured_task()', number=times, globals=dict(measured_task=task)) / times
    # print('Method: %d; T: %0.2f; Error: %s; Eps: %0.2f; Delta: %0.5f; Coverage: %0.2f%%; Match: %0.2f%%; R2: %0.4f; Std: %0.4f; Avg.time: %fs' %
    #      (method, T, err_name, eps, delta, cover_percent, match_percent, r2, stdev, time))
    return delta, cover_percent, match_percent, r2, stdev, time, other


def test_h_prediction(series_real, h_real, series, h_low, h_up):
    l, r = np.searchsorted(series_real, series[0]), np.searchsorted(series_real, series[-1], side='right')
    series_real = series_real[l:r]
    cover_percent = len(series_real) / len(h_real) * 100
    h_real = h_real[l:r]
    h_low, h_up = np.interp(series_real, series, h_low), np.interp(series_real, series, h_up)
    match_percent = calculate_match_percent(h_real, h_low, h_up)
    h_cent = (h_low + h_up) / 2
    r2 = r2_score(h_real, h_cent)
    stdev = np.sqrt(np.sum((h_up - h_low)**2) / (len(h_low) - 1))
    return match_percent, cover_percent, r2, stdev, series_real, np.abs(h_real- h_cent)


def autolabel(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.5f' % height,
                ha='center', va='bottom')

def autolabel_inside(ax, rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%0.5f' % height,
                ha='center', va='top')


def method_analyzer(T, delta, error, ignore_second=False):
    # params : cover_percent, match_percent, r2, stdev, time, h_series, abs(h_real - h_pred)
    delta, *params1 = run_m1(T=T, eps=0.5, delta=delta, err_gen=error)
    if not ignore_second:
        delta, *params2 = run_m2(T=T, eps=0.5, delta=delta, err_gen=error)
    depth = int(np.floor(np.log2(T / delta))) - 1
    delta, *params3 = run_m3(T=T, eps=0.5, depth=depth, J=5, err_gen=error)
    params = [params1, params3] if ignore_second else [params1, params2, params3]
    
    # Show coverage and matching
    fig1, ax1 = plt.subplots(figsize=(12,5))
    index = np.arange(2)
    bar_width = 0.35 if ignore_second else 0.25
    opacity = 0.4
    rects1 = ax1.bar(index, params[0][:2], bar_width,
                     alpha=opacity,
                     color='b',
                     label='Метод 1')
    autolabel_inside(ax1, rects1)
    if not ignore_second:
        rects2 = ax1.bar(index + bar_width, params[1][:2], bar_width,
                     alpha=opacity,
                     color='r',
                     label='Метод 2')
        autolabel_inside(ax1, rects2)
        rects3 = ax1.bar(index + 2 * bar_width, params[-1][:2], bar_width,
                     alpha=opacity,
                     color='g',
                     label='Метод 3')
    else:
        rects3 = ax1.bar(index + bar_width, params[-1][:2], bar_width,
                         alpha=opacity,
                         color='g',
                         label='Метод 3')
    autolabel_inside(ax1, rects3)
    ax1.set_xlabel('Статистики')
    ax1.set_ylabel('%')
    ax1.set_title('Статистики покрытия и попадания в доверительные интервалы %s' % (error.__name__))
    if ignore_second:
        ax1.set_xticks(index + bar_width)
    else:
        ax1.set_xticks(index + 1.5 * bar_width)
    ax1.set_xticklabels(('Покрытие', 'Попадание'))
    ax1.legend(loc=0)
    ax1.grid(False)
    fig1.tight_layout()
    
    # Show r2, stdev, time
    fig2, (ax21, ax22, ax23) = plt.subplots(1, 3, figsize=(12,5))
    index = np.arange(len(params))
    xlabels = ('Метод 1', 'Метод 2', 'Метод 3') if not ignore_second else ('Метод 1', 'Метод 3')
    bar_width = 0.5
    ax21.set_title('$R^2$')
    ax21.scatter(index, [p[2] for p in params], marker='^', s=40, color='r')
    ax21.set_xticks(index)
    ax21.set_xticklabels(xlabels)
    ax22.set_title('Std')
    ax22.grid(False)
    rects2 = ax22.bar(index, [p[3] for p in params], bar_width,
                      alpha=opacity,
                      color='b')
    autolabel(ax22, rects2)
    ax22.set_xticks(index + 0.5 * bar_width)
    ax22.set_xticklabels(xlabels)
    ax23.set_title('Время работы')
    ax23.grid(False)
    rects3 = ax23.bar(index, [p[4] for p in params], bar_width,
                      alpha=opacity,
                      color='b')
    autolabel(ax23, rects3)
    ax23.set_ylabel('s')
    ax23.set_xticks(index + 0.5 * bar_width)
    ax23.set_xticklabels(xlabels)
    fig2.tight_layout()
    # Show residual
    fig3, ax3 = plt.subplots(figsize=(12,5))
    if not ignore_second:
        h_pair1, h_pair2, h_pair3 = (p[-1] for p in params)
    else:
        h_pair1, h_pair3 = (p[-1] for p in params)
    ax3.set_title('Невязка: abs(h_real - h_central)')
    ax3.plot(h_pair1[0], h_pair1[1], 'r', label='$|\Delta h|_1$')
    if not ignore_second:
        ax3.plot(h_pair2[0], h_pair2[1], 'g', label='$|\Delta h|_2$')
    ax3.plot(h_pair3[0], h_pair3[1], 'b', label='$|\Delta h|_3$')
    ax3.legend(loc=0)
    fig3.tight_layout()

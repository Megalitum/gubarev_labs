from pylab import *
from scipy import optimize, linalg
import seaborn as sns
import itertools
from model import ObservableSystem

observable = ObservableSystem()

# TODO: add Hankel's matrix method
# TODO: add more sophisticated initial point prediction
# TODO: add smoothing factor (e.g. build smoothing spline and call optimization on first approximation)
# TODO: add current error roots calculation to predict most relevant harmonic component

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
            approx = ObservableSystem.unit_response(domain, vec, delta, False)
            return np.amax(abs(noisy - approx)) / domain.shape[0]
        return cost_func
    elif type == 'l2':
        def cost_func(vec):
            approx, jac = ObservableSystem.unit_response(domain, vec, delta, True)
            return sum((noisy - approx)**2), 2 * ((approx - noisy) * jac).sum(axis=1)
        return cost_func
    else:
        raise NotImplemented()

def starting_points():
    yield from itertools.product([-1, 1], [-1, 1], [0], [4])


def perform_approximaiton(domain, max_q, type='l2'):
    minimizer_kwargs = {"method": "SLSQP", "jac": type == 'l2',
                        "bounds": ((None, None), (None, None), (0, 6), (-10, 10))}
    noisy = observable(domain)
    for q in range(max_q):
        cost_func = generate_cost_func(domain, noisy, type)
        best_sol = None
        for start_point in starting_points():
            print('>', best_sol.fun if best_sol is not None else None)
            sol = optimize.basinhopping(cost_func, start_point, minimizer_kwargs=minimizer_kwargs)
            if best_sol is None or sol.fun < best_sol.fun:
                best_sol = sol
        noisy -= ObservableSystem.response(domain, best_sol.x, observable.delta)
        print("Status: ", best_sol.message)
        print("f, f':", cost_func(best_sol.x))
        yield best_sol

def Gankels_matrix(values):
    values = atleast_1d(values)
    assert(values.size % 2 == 0)
    k = values.size // 2
    m = zeros((k, k))
    for i in range(k):
        m[i, :] = values[i:i+k]
    return m

def svd_r(matrix, q):
    qr, sr, vr_t = linalg.svd(matrix)
    reduced_sr = sr[0:q]
    reduced_m_sr = diag(reduced_sr)
    reduced_qr = qr[:,0:q]
    reduced_vr_t = vr_t[0:q, :]
    return reduced_qr, reduced_sr, reduced_vr_t

def matrix_reduction(values, q, delta):
    matrix = Gankels_matrix(values)
    reduced_qr, reduced_ser, reduced_vr_t = svd_r(matrix, q)
    # g1, g2 = generate_params(reduced_qr)
    # b, residues, rank, s = scipy.linalg.lstsq(g1, g2)
    b_log_eigenvalues = log(linalg.eigvals(b)) / delta
    return b_log_eigenvalues

max_q = 8
N = 500
domain = arange(1, N + 1, 1)
result = None
norm_type = 'l2'
for q, solution in enumerate(perform_approximaiton(domain, max_q, norm_type)):
    if result is None:
        result = solution.x[:, newaxis]
    else:
        result = np.append(result, solution.x[:, newaxis], axis=1)
    y_real = observable(domain)
    y_appr = ObservableSystem.response(domain, result.ravel(), observable.delta)
    plot(domain, y_real, '-r', label='real')
    plot(domain, y_appr, '-b', label='approximation')
    legend(loc='upper right')
    title(str(q + 1))
    figure()
    plot(domain, y_real - y_appr, 'g')
    title('Noise %d' % (q+1))
    savefig('lab1_out/noise_%s.png' % (q+1))
    #show()

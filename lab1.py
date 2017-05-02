import itertools

from pylab import *
from scipy import optimize

from reduction.model import ObservableSystem, SimplifiedObservableSystem
from reduction.reductors import ResponseFitReductor


def test_svd(method):
    from reduction.reductors import SVDReductor
    observable = SimplifiedObservableSystem(path='lab1_data')
    N = 250
    domain = arange(1, N + 1, 1)
    resp = observable(domain)
    reductor = SVDReductor(observable.delta)
    approx_observable = reductor.generate_model(method, domain, resp, 14)
    figure()
    scatter(observable.all_eigenvalues.real, observable.all_eigenvalues.imag, c='r', marker='+')
    scatter(real(approx_observable.all_eigenvalues), imag(approx_observable.all_eigenvalues), marker='x')
    figure()
    plot(domain, resp, 'r')
    plot(domain, approx_observable(domain))
    show()


def test_response_fit(norm_type):
    observable = ObservableSystem(path='lab1_data')
    max_q = 14
    N = 250
    domain = arange(1, N + 1, 1)
    y_real = observable(domain)

    def callback(x, response, q):
        val = -x[2] + 1j * x[3]
        indices = np.argmin(np.abs(observable.eigenvalues - val))
        f_coef = observable.f_cc[indices], observable.f_ss[indices]
        print("Values: {:20f} {:20f} {:20f}".format(val, x[0], x[1]))
        print("Real v: {:20f} {:20f} {:20f}".format(observable.eigenvalues[indices], f_coef[0], f_coef[1]))
    reductor = ResponseFitReductor(observable.delta,
                                   ((-1, 1), (-1, 1), (-0.5, 6), (-0.5, 8)),
                                   callback=callback)
    reduced_model = reductor.generate_model(norm_type, domain, y_real, max_q)
    figure()
    scatter(observable.all_eigenvalues.real, observable.all_eigenvalues.imag, c='r', marker='+')
    scatter(reduced_model.all_eigenvalues.real, reduced_model.all_eigenvalues.imag, marker='x')
    figure()
    plot(domain, observable(domain), 'r')
    plot(domain, reduced_model(domain))
    show()


if __name__ == '__main__':
    #test_response_fit('basinhopping_l2')
    test_svd('lstsq')
import itertools

from pylab import *
from scipy import optimize

from reduction.model import ObservableSystem, SimplifiedObservableSystem
from reduction.reductors import ResponseFitReductor



def test_svd():
    from reduction.reductors import SVDReductor
    observable = SimplifiedObservableSystem()
    N = 500
    domain = arange(1, N + 1, 1)
    resp = observable(domain)
    reductor = SVDReductor(observable.delta)
    approx_observable = reductor.generate_model('solve', domain, resp, 10)
    scatter(observable.eigenvalues.real, observable.eigenvalues.imag, c='r', marker='+')
    scatter(real(approx_observable.eigenvalues), imag(approx_observable.eigenvalues), marker='x')
    figure()
    plot(domain, resp, 'r')
    plot(domain, approx_observable(domain))
    show()

def test_response_fit():
    observable = ObservableSystem()
    max_q = 8
    N = 500
    domain = arange(1, N + 1, 1)
    norm_type = 'l2'
    y_real = observable(domain)

    def callback(response, q):
        # figure()
        # plot(domain, y_real, '-r', label='real')
        # plot(domain, response, '-b', label='approximation')
        # legend(loc='upper right')
        # title(str(q + 1))
        # figure()
        # plot(domain, y_real - response, 'g')
        # title('Noise %d' % (q+1))
        # savefig('lab1_out/noise_%s.png' % (q+1))
        pass
    reductor = ResponseFitReductor(observable.delta, callback=callback)
    reduced_model = reductor.generate_model(norm_type, domain, y_real, max_q)
    figure()
    scatter(observable.eigenvalues.real, observable.eigenvalues.imag, color='r')
    scatter(reduced_model.eigenvalues.real, reduced_model.eigenvalues.imag, marker='x')
    show()

if __name__ == '__main__':
    test_response_fit()
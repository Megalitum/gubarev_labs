import itertools

from pylab import *
import seaborn as sns
from scipy import optimize
matplotlib.rcParams['figure.figsize'] = (15, 10)

from reduction.model import ObservableSystem, SimplifiedObservableSystem
from reduction.reductors import ResponseFitReductor
from reduction.stats import general_test


def test_svd():
    from reduction.reductors import SVDReductor
    observable = ObservableSystem(path='lab1_data')
    reductor = SVDReductor(observable.delta)
    general_test(observable, reductor, 500, 20)


def test_response_fit():
    observable = ObservableSystem(path='lab1_data')
    fig, ax = subplots()
    converging = {}
    def callback(method, x, response, q):
        ax.plot(response, '--', label=f"{q}", markersize=2)
        if method not in converging:
            converging[method] = []
        converging[method].append((q, response))
    reductor = ResponseFitReductor(observable.delta,
                                   ((-1, 1), (-1, 1), (-0.5, 6), (-0.5, 8)),
                                   callback=callback)
    general_test(observable, reductor, 250, 20)
    for key, value in converging.items():
        fig, ax = subplots()
        alpha_mult = 1 / len(value) / 2
        for i, (q, resp) in enumerate(value):
            ax.plot(resp, 'g', alpha=(i+1) * alpha_mult, linewidth=1)
        ax.plot(value[-1][1], 'g', linewidth=1)
        ax.plot(observable(arange(1, 250)), 'r', linewidth=1)
        fig.savefig('lab1_out/' + key + '.png')


if __name__ == '__main__':
    test_response_fit()
    test_svd()
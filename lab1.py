import itertools

from pylab import *
import seaborn as sns
from scipy import optimize
matplotlib.rcParams['figure.figsize'] = (12, 8)

from reduction.model import ObservableSystem, SimplifiedObservableSystem
from reduction.reductors import ResponseFitReductor
from reduction.stats import general_test


def test_svd():
    from reduction.reductors import SVDReductor
    observable = ObservableSystem(path='lab1_data')
    reductor = SVDReductor(observable.delta)
    tick_count = 250
    general_test(observable, reductor, tick_count, 20)
    domain = arange(1, 1 + tick_count)
    resp_ref = observable(domain)
    errors = []
    for reduced_dim in range(1, 11):
        reduced_observable = reductor.generate_model('lstsq', domain, resp_ref, reduced_dim, verbose=False)
        resp_est = reduced_observable(domain)
        errors.append([reduced_dim, np.linalg.norm(resp_ref - resp_est),
                       np.linalg.norm(resp_ref - resp_est, inf)])
    fig, axes = subplots(2, 1, figsize=(10, 5), sharex=True)
    errors = np.array(errors, dtype=np.float64)
    axes[0].set_title(r'$||\cdot||_2$')
    axes[0].set_ylabel('error')
    axes[0].set_xlabel('Q')
    axes[0].plot(errors[:, 0], errors[:, 1], c='black')
    axes[1].set_ylabel('error')
    axes[1].set_xlabel('Q')
    axes[1].set_title(r'$||\cdot||_{\infty}$')
    axes[1].plot(errors[:, 0], errors[:, 2], c='black')
    for ax in axes:
        ax.set_ylim((0, ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig('lab1_out/' + 'svd_lstsq' + '_error.png')



def test_response_fit():
    observable = ObservableSystem(path='lab1_data')
    converging = {}
    tick_count = 250
    domain = arange(1, 1 + tick_count)
    resp_ref = observable(domain)
    def callback(method, x, response, q):
        if method not in converging:
            converging[method] = {}
        converging[method][q + 1] = (np.linalg.norm(resp_ref - response),
                                     np.linalg.norm(resp_ref - response, inf))
    reductor = ResponseFitReductor(observable.delta,
                                   ((-1, 1), (-1, 1), (-0.5, 6), (-0.5, 8)),
                                   callback=callback)
    general_test(observable, reductor, tick_count, 20)
    for i, (key, value) in enumerate(converging.items()):
        fig, axes = subplots(2, 1, figsize=(10, 5), sharex=True)
        errors = np.array(list([q, l2, linf] for q, (l2,linf) in value.items()), dtype=np.float64)
        axes[0].set_title(r'$||\cdot||_2$')
        axes[0].set_ylabel('error')
        axes[0].set_xlabel('Q')
        axes[0].plot(errors[:,0], errors[:,1], c='black')
        axes[1].set_ylabel('error')
        axes[1].set_xlabel('Q')
        axes[1].set_title(r'$||\cdot||_{\infty}$')
        axes[1].plot(errors[:,0], errors[:,2], c='black')
        for ax in axes:
            ax.set_ylim((0, ax.get_ylim()[1]))
        fig.tight_layout()
        fig.savefig('lab1_out/' + key + '_error.png')


if __name__ == '__main__':
    test_response_fit()
    test_svd()
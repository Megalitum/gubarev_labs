from timeit import Timer
import sklearn.metrics.regression as reg_metrics
from pylab import *


def general_test(observable, reductor, tick_count, reduced_dim, methods=None):
    reductor_name = type(reductor).__name__
    domain = arange(1, tick_count + 1)
    resp_ref = observable(domain)
    with open("lab1_out/" + reductor_name + ".log", "w") as f:
        for method in methods if methods is not None else reductor.available_methods():
            full_method_name = "{}#{}".format(reductor_name, method)
            print(full_method_name)
            reduced_observable = reductor.generate_model(method, domain, resp_ref, reduced_dim, verbose=True)
            eigenvalues_fit_report(observable.eigenvalues, reduced_observable.eigenvalues)
            savefig("lab1_out/" + full_method_name + ".eigs.png")
            resp_est = reduced_observable(domain)
            timer = Timer(lambda: reductor.generate_model(method, domain, resp_ref, reduced_dim))
            fit_report_results = response_fit_report(resp_ref, resp_est)
            savefig("lab1_out/" + full_method_name + ".resp.png")
            fit_report_results['time'] = timer.timeit(1) / 1
            f.writelines(("""Method {method}
    Avg.time: {time}
    R2: {r2}
    MAE: {mae}
    MSE: {mse}\n""".format(method=method,**fit_report_results)))



def eigenvalues_fit_report(eig_ref, eig_est):
    figure()
    xlabel('Re')
    ylabel('Im')
    scatter(np.stack((eig_ref.real, eig_ref.real)),
            np.stack((eig_ref.imag, -eig_ref.imag)), c='black', marker='o', label='input')
    scatter(np.stack((eig_est.real, eig_est.real)),
            np.stack((eig_est.imag, -eig_est.imag)), c='black', marker='x', label='output')
    legend()


def response_fit_report(resp_ref, resp_est):
    figure()
    plot(resp_ref, c='black', linewidth=1, label='$y_k$')
    plot(resp_est, c='black', linestyle='--', linewidth=1, label='$\\tilde y_k$')
    legend()
    results = {}
    results['r2'] = reg_metrics.r2_score(resp_ref, resp_est)
    results['mae'] = reg_metrics.mean_absolute_error(resp_ref, resp_est)
    results['mse'] = reg_metrics.mean_squared_error(resp_ref, resp_est)
    return results
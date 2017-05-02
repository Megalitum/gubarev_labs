import scipy.linalg as linalg
import scipy.optimize as optimize
import numpy as np
import itertools

from .model import ObservableSystem, SimplifiedObservableSystem


class Reductor:
    def __init__(self, delta):
        self.delta = delta

    def generate_model(self, method, domain, response, q):
        func_name = 'generate_%s' % (method)
        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            return getattr(self, func_name)(domain, response, q)


class MyBounds(object):
    def __init__(self, xmax=None, xmin=None):
        if xmin is None:
            xmin = [-1.1, -1.1]
        if xmax is None:
            xmax = [1.1, 1.1]
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class SVDReductor(Reductor):
    @staticmethod
    def calculate_gamma_matrix(response, q):
        assert response.ndim == 1
        first_row_size = response.size // 2
        h_matrix = linalg.hankel(response[0:first_row_size], response[-first_row_size:])
        u, sigma, _ = linalg.svd(h_matrix)
        return u[:, :q]

    def generate_lstsq(self, domain, response, q):
        gamma_matrix = SVDReductor.calculate_gamma_matrix(response, q)
        a_exp, residues, rank, s = linalg.lstsq(gamma_matrix[:-1], gamma_matrix[1:])
        # print('LS results: ', residues, rank, s)
        params = np.log(linalg.eigvals(a_exp)) / self.delta
        return SimplifiedObservableSystem(eigenvalues=params, delta=self.delta)


class ResponseFitReductor(Reductor):
    def __init__(self, delta, bounds, callback=None):
        super().__init__(delta)
        self.bounds = bounds
        if callback is not None:
            assert callable(callback), "Given callback is not callable."
        self.callback = callback

    @staticmethod
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
                return np.sum((noisy - approx) ** 2), 2 * ((approx - noisy) * jac).sum(axis=1)
            return cost_func
        else:
            raise NotImplemented()

    def generate_basinhopping_l2(self, domain, response, max_q):
        norm_type = "l2"
        result = np.ndarray(shape=(4,0))
        noisy = response[:]
        basic_minimizer_args = {"method": "SLSQP", "jac": True}
        last_sol = None
        for q in range(max_q // 2):
            cost_func = ResponseFitReductor.generate_cost_func(domain, noisy,
                                                               norm_type, self.delta)
            best_sol = None
            for start_point in itertools.combinations_with_replacement([-1, 1], 2):
                start_point = np.append(start_point, [0,0] if last_sol is None else last_sol[2:4])
                minimizer_kwargs = basic_minimizer_args.copy()
                minimizer_kwargs["bounds"] = ((max(self.bounds[0][0], start_point[0] - 0.2),
                                               min(self.bounds[0][1], start_point[0] + 0.2)),
                                              (max(self.bounds[1][0], start_point[1] - 0.2),
                                               min(self.bounds[1][1], start_point[1] + 0.2)),
                                              self.bounds[2],
                                              self.bounds[3])
                sol = optimize.basinhopping(cost_func, start_point, niter_success=50, seed=2442,
                                            minimizer_kwargs=minimizer_kwargs,
                                            accept_test=MyBounds(xmin=[b[0] for b in minimizer_kwargs["bounds"]],
                                                                 xmax=[b[1] for b in minimizer_kwargs["bounds"]]))
                if best_sol is None or sol.fun < best_sol.fun:
                    best_sol = sol
            last_sol = best_sol.x
            print("Found solution: ", last_sol)
            noisy -= ObservableSystem.response(domain, last_sol, self.delta)
            result = np.append(result, best_sol.x[:, np.newaxis], axis=1)
            y_appr = ObservableSystem.response(domain, result.ravel(), self.delta)
            if self.callback is not None:
                self.callback(last_sol, y_appr, q*2)
            print("Status: ", best_sol.message)
            print("f, f':", cost_func(best_sol.x))
        eigenvalues = -result[2] + result[3] * 1j
        f_cc = result[0]
        f_ss = result[1]
        return ObservableSystem(eigenvalues=eigenvalues, f_params=(f_cc, f_ss), delta=self.delta)


    def generate_diff_evolution_l_inf(self, domain, response, max_q):
        norm_type = "l_inf"
        result = np.ndarray(shape=(4,0))
        noisy = response[:]
        for q in range(max_q // 2):
            cost_func = ResponseFitReductor.generate_cost_func(domain, noisy,
                                                               norm_type, self.delta)
            best_sol = None
            for f_c, f_s in itertools.combinations_with_replacement([-1, 1], 2):
                bounds = ((max(self.bounds[0][0], f_c - 0.2), min(self.bounds[0][1], f_c + 0.2)),
                          (max(self.bounds[1][0], f_s - 0.2), min(self.bounds[1][1], f_s + 0.2)),
                          self.bounds[2],
                          self.bounds[3])
                sol = optimize.differential_evolution(cost_func, bounds, strategy='rand1exp', polish=True,
                                                      seed=142, popsize=100, maxiter=10000, mutation=(0.5, 1.5))
                if best_sol is None or sol.fun < best_sol.fun:
                    best_sol = sol
            last_sol = best_sol.x
            print("Found solution: ", last_sol)
            noisy -= ObservableSystem.response(domain, last_sol, self.delta)
            result = np.append(result, best_sol.x[:, np.newaxis], axis=1)
            y_appr = ObservableSystem.response(domain, result.ravel(), self.delta)
            if self.callback is not None:
                self.callback(last_sol, y_appr, q*2)
            print("Status: ", best_sol.message)
            print("f:", cost_func(best_sol.x))
        eigenvalues = -result[2] + result[3] * 1j
        f_cc = result[0]
        f_ss = result[1]
        return ObservableSystem(eigenvalues=eigenvalues, f_params=(f_cc, f_ss), delta=self.delta)

    def generate_brute_l_inf(self, domain, response, max_q):
        norm_type = "l_inf"
        result = np.ndarray(shape=(4,0))
        noisy = response[:]
        for q in range(max_q // 2):
            cost_func = ResponseFitReductor.generate_cost_func(domain, noisy,
                                                               norm_type, self.delta)
            best_sol = None
            for f_c, f_s in itertools.combinations_with_replacement([-1, 1], 2):
                bounds = ((max(self.bounds[0][0], f_c - 0.2), min(self.bounds[0][1], f_c + 0.2)),
                          (max(self.bounds[1][0], f_s - 0.2), min(self.bounds[1][1], f_s + 0.2)),
                          self.bounds[2],
                          self.bounds[3])
                sol = optimize.brute(cost_func, bounds, Ns=20, finish=None)
                if best_sol is None or cost_func(sol) < cost_func(best_sol):
                    best_sol = sol
            last_sol = best_sol
            print("Found solution: ", last_sol)
            noisy -= ObservableSystem.response(domain, last_sol, self.delta)
            result = np.append(result, best_sol[:, np.newaxis], axis=1)
            y_appr = ObservableSystem.response(domain, result.ravel(), self.delta)
            if self.callback is not None:
                self.callback(last_sol, y_appr, q*2)
            print("f:", cost_func(best_sol))
        eigenvalues = -result[2] + result[3] * 1j
        f_cc = result[0]
        f_ss = result[1]
        return ObservableSystem(eigenvalues=eigenvalues, f_params=(f_cc, f_ss), delta=self.delta)
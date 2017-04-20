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


class SVDReductor(Reductor):
    @staticmethod
    def calculate_gamma_matrix(response, q):
        assert response.ndim == 1
        first_row_size = response.size // 2
        h_matrix = linalg.hankel(response[0:first_row_size], response[-first_row_size:])
        u, sigma, _ = linalg.svd(h_matrix)
        return u[:, :q]

    def generate_solve(self, domain, response, q):
        gamma_matrix = SVDReductor.calculate_gamma_matrix(response, q)
        a_exp = linalg.solve(gamma_matrix[:-1].T @ gamma_matrix[:-1], gamma_matrix[:-1].T @ gamma_matrix[1:])
        params = np.log(linalg.eigvals(a_exp)) / self.delta
        return SimplifiedObservableSystem(params)

    def generate_lstsq(self, domain, response, q):
        gamma_matrix = SVDReductor.calculate_gamma_matrix(response, q)
        a_exp, residues, rank, s = linalg.lstsq(gamma_matrix[:-1], gamma_matrix[1:])
        # print('LS results: ', residues, rank, s)
        params = np.log(linalg.eigvals(a_exp)) / self.delta
        return SimplifiedObservableSystem(params)

# TODO: add more sophisticated initial point prediction


class ResponseFitReductor(Reductor):
    def __init__(self, delta, callback=None):
        super().__init__(delta)
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
                approx += ObservableSystem.unit_response(domain, vec * np.array([1,1,1,-1]), delta, False)
                return np.amax(abs(noisy - approx)) / domain.shape[0]
            return cost_func
        elif type == 'l2':
            def cost_func(vec):
                approx, jac = ObservableSystem.unit_response(domain, vec, delta, True)
                approx2, jac2 = ObservableSystem.unit_response(domain, vec* np.array([1,1,1,-1]), delta, True)
                approx += approx2
                jac += jac2 * np.array([[1],[1],[1],[-1]])
                return np.sum((noisy - approx) ** 2), 2 * ((approx - noisy) * jac).sum(axis=1)
            return cost_func
        else:
            raise NotImplemented()

    def perform_approximaiton(self, domain, response, max_q, norm_type):
        noisy = response[:]
        basic_minimizer_args = {"method": "SLSQP", "jac": norm_type == 'l2'}
        last_sol = None
        for q in range(max_q // 2):
            cost_func = ResponseFitReductor.generate_cost_func(domain, noisy,
                                                               norm_type, self.delta)
            best_sol = None
            for start_point in itertools.combinations_with_replacement([-1, 1], 2):
                start_point = np.append(start_point, [0,0] if last_sol is None else last_sol[2:4])
                minimizer_kwargs = basic_minimizer_args.copy()
                minimizer_kwargs["bounds"] = ((start_point[0] - 0.2, start_point[0] + 0.2),
                                              (start_point[1] - 0.2, start_point[1] + 0.2), (0, 6), (-10, 10))
                sol = optimize.basinhopping(cost_func, start_point, minimizer_kwargs=minimizer_kwargs)
                if best_sol is None or sol.fun < best_sol.fun:
                    best_sol = sol
            last_sol = best_sol.x
            noisy -= ObservableSystem.response(domain, last_sol, self.delta)
            noisy -= ObservableSystem.response(domain, last_sol * np.array([1,1,1,-1]), self.delta)
            print("Status: ", best_sol.message)
            print("f, f':", cost_func(best_sol.x))
            yield best_sol

    def generate_l2(self, domain, response, max_q):
        result = None
        norm_type = "l2"
        for q, solution in enumerate(self.perform_approximaiton(domain, response, max_q, norm_type)):
            if result is None:
                result = solution.x[:, np.newaxis]
            else:
                result = np.append(result, solution.x[:, np.newaxis], axis=1)
            y_appr = ObservableSystem.response(domain, result.ravel(), self.delta)
            if self.callback is not None:
                self.callback(y_appr, q*2)
        eigenvalues = np.concatenate((-result[2] + result[3] * 1j, -result[2] - result[3] * 1j))
        f_cc = np.tile(result[0], 2)
        f_ss = np.tile(result[1], 2)
        return ObservableSystem(eigenvalues, (f_cc, f_ss))
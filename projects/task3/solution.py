"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor

from scipy.stats import norm

from numpy import random

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

SEED = 0

np.random.seed(SEED)

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""

        # self.f_kernel = Matern(nu=2.5, length_scale_bounds="fixed")
        self.f_kernel = ConstantKernel(constant_value=0.5, constant_value_bounds='fixed') * \
                        Matern(length_scale=0.5, nu=2.5, length_scale_bounds='fixed')
        self.f_GP = GaussianProcessRegressor(kernel=self.f_kernel, alpha=0.15 ** 2, random_state=SEED)

        # self.v_kernel = Matern(nu=2.5, length_scale_bounds="fixed")
        self.v_kernel = ConstantKernel(constant_value=4, constant_value_bounds='fixed') + \
                        ConstantKernel(constant_value=np.sqrt(2), constant_value_bounds='fixed') * \
                        Matern(length_scale=0.5, nu=2.5, length_scale_bounds='fixed')
        self.v_GP = GaussianProcessRegressor(kernel=self.v_kernel, alpha=0.0001 ** 2, random_state=SEED)

        self.x_data = np.array([]).reshape(-1, 1)
        self.f_data = np.array([]).reshape(-1, 1)
        self.v_data = np.array([]).reshape(-1, 1)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        if self.x_data.size == 0:
            rec = 10 * np.random.rand()  # random float between 0.0 and 10.0 TODO max 4?
        else:
            rec = self.optimize_acquisition_function()
            # rec = np.array([[rec]]) # No longer needed as fixed in main function
        return rec

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        beta = 0.5

        mu_f, std_f = self.f_GP.predict(x, return_std=True)
        mu_v, std_v = self.f_GP.predict(x, return_std=True)

        return np.squeeze(mu_f + beta * std_f) * np.squeeze(
            norm.cdf((mu_v + SAFETY_THRESHOLD) / std_v))

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        self.x_data = np.vstack((self.x_data, x))
        self.f_data = np.vstack((self.f_data, f))
        self.v_data = np.vstack((self.v_data, v))

        self.f_GP.fit(self.x_data, self.f_data)
        self.v_GP.fit(self.x_data, self.v_data)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        masked = self.f_data
        masked[self.v_data > SAFETY_THRESHOLD] = np.NINF
        opt_idx = np.argmax(masked)
        opt = self.x_data[opt_idx, 0]
        return opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape TODO DEACTIVATED FOR NOW
        # assert x.shape == (1, DOMAIN.shape[0]), \
        #    f"The function next recommendation must return a numpy array of " \
        #    f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

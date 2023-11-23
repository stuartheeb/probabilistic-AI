"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, DotProduct, RBF
import matplotlib.pyplot as plt
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
STD_DEV_V = 0.0001
STD_DEV_F = 0.15

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # v_kernel = DotProduct(sigma_0=0) + Matern(nu=2.5)
        v_kernel = ConstantKernel(constant_value=4, constant_value_bounds='fixed') \
        + DotProduct(sigma_0=0) + Matern(nu=2.5, length_scale=0.5, length_scale_bounds='fixed')
        self.v_gp = GaussianProcessRegressor(kernel=v_kernel, n_restarts_optimizer=10, alpha=2*STD_DEV_V**2)

        f_kernel = Matern(nu=2.5) #TODO: try RBF kernel
        # f_kernel = RBF(length_scale=1)
        self.f_gp = GaussianProcessRegressor(kernel=f_kernel, n_restarts_optimizer=10, alpha=2*STD_DEV_F**2)

        self.x = np.array([])
        self.y_f = np.array([])
        self.y_v = np.array([])
        np.random.seed(0)

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
        while(True):
            x_opt = self.optimize_acquisition_function()
            x_opt = np.array([[x_opt]])
            self.last_recommended = x_opt
            return x_opt

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
        #UCB
        beta = 4.0
        gamma = 100
        mean, std = self.f_gp.predict(x.reshape(-1,1), return_std=True)
        mean_v, std_v = self.v_gp.predict(x.reshape(-1,1), return_std=True)
        acc = mean + beta * std - gamma*np.max(mean_v + beta*std_v - SAFETY_THRESHOLD, 0) # Calculate UCB.
        # while(True):
        #     rand = np.random.rand(1)*10
        #     acc, acc_std = self.v_gp.predict(rand.reshape(-1,1), return_std=True)
        #     if not(acc + 2*acc_std > SAFETY_THRESHOLD):
        #         return rand
        return acc

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
        self.y_f = np.append(self.y_f, f)
        self.y_v = np.append(self.y_v, v)
        self.x = np.append(self.x, x)
        self.f_gp.fit(self.x.reshape(-1,1), self.y_f.reshape(-1,1))
        self.v_gp.fit(self.x.reshape(-1,1), self.y_v.reshape(-1,1))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        def objective(x):
            v_pred = self.v_gp.predict(x.reshape(-1,1))
            if(v_pred > SAFETY_THRESHOLD):
                return -self.f_gp.predict(x.reshape(-1,1))[0] + 1e3 * (v_pred - SAFETY_THRESHOLD)
            return -self.f_gp.predict(x.reshape(-1,1))[0]

        f_values = []
        x_values = []
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            x1 = x0.reshape(-1,1)
            result = fmin_l_bfgs_b(objective, x0=x1, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()
        return x_opt

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        self.figure = plt.figure(figsize=(12, 8))
        n = np.linspace(*DOMAIN[0], 500)
        f_mean, f_std = self.f_gp.predict(n.reshape(-1,1), return_std=True)
        v_mean, v_std = self.v_gp.predict(n.reshape(-1,1), return_std=True)
        f_last_recommended = self.f_gp.predict(self.last_recommended)
        f_true = np.vectorize(f)(n)
        v_true = np.vectorize(v)(n)
        plt.plot(n, f_mean, label='f mean')
        plt.plot(n, f_true, label='f true', color='grey', alpha=0.7, linestyle='dashed')
        plt.plot(n, v_true, label='v true', color='black', alpha=0.7, linestyle='dashed')
        plt.plot(n, v_mean, label='v mean', color='red')
        plt.fill_between(n, f_mean - 2 * f_std, f_mean + 2 * f_std, alpha=0.2)
        plt.fill_between(n, v_mean- 2 * v_std, v_mean + 2 * v_std, alpha=0.2)
        plt.scatter(self.x, self.y_f, label='f data')
        plt.scatter(self.last_recommended, f_last_recommended, label='last recommended', color='red')
        plt.legend(loc='upper left')
        plt.show()

    


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---
def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init
    
def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 3 + 2*np.sin(x*2)


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])





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
        agent.plot()
        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(loc=0, scale=STD_DEV_F)
        cost_val = v(x) + np.random.normal(loc=0, scale=STD_DEV_V)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    agent.plot()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()

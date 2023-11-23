"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import gpytorch
import torch
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, type, iterations=50):
        # super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        super().__init__(train_x, train_y, likelihood)
        self.training_iter = iterations
        self.model_type = type

        k = gpytorch.kernels
        if type == 'f':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = k.ScaleKernel(k.MaternKernel(nu=2.5))
            # self.covar_module = k.MaternKernel(nu=2.5)
        elif type == 'v':
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = k.AdditiveKernel(k.LinearKernel(), k.MaternKernel(nu=2.5))
        else:
            raise Exception(f"model hast to be 'f' of 'v' but is {type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale

    @length_scale.setter
    def length_scale(self, value):
        self.covar_module.base_kernel.lengthscale = value



class BOpt(torch.nn.Module):
    """Abstract Bayesian Optimization class. ."""

    def __init__(self, gp, x, beta=2.0):
        super().__init__()
        self.gp = gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.x = x
        self.update_acquisition_function(x)

    def update_gp(self, new_inputs, new_targets):
        """Update GP with new points."""
        new_inputs, new_targets = torch.FloatTensor(new_inputs), torch.FloatTensor(new_targets)
        if len(new_targets.shape) == 2:
            new_targets = new_targets[0]
        inputs = torch.cat((self.gp.train_inputs[0], new_inputs[0]), dim=0)
        targets = torch.cat((self.gp.train_targets, new_targets), dim=-1)
        self.gp.set_train_data(inputs, targets, strict=False)
        self.update_acquisition_function(new_inputs)

    def get_best_value(self):
        idx = self.gp.train_targets.argmax()
        if len(self.gp.train_targets) == 1:
            xmax, ymax = self.gp.train_inputs[idx], self.gp.train_targets[idx]
        else:
            xmax, ymax = self.gp.train_inputs[0][idx], self.gp.train_targets[idx]
        return xmax, ymax

    def update_acquisition_function(self, x):
        raise NotImplementedError

    @property
    def acquisition_function(self):
        return self._acquisition_function

    def forward(self):
        """Call the algorithm. """
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y = self.acquisition_function
            max_id = torch.argmax(y)
            next_point = self.x[[[max_id]]]
        return next_point


class GPUCB(BOpt):
    def __init__(self, gp, x, beta=2.0):
        self.beta = beta
        super().__init__(gp, x)

    def update_acquisition_function(self, x):
        pred = self.gp(torch.FloatTensor(x))
        ucb = pred.mean + self.beta * pred.stddev  # Calculate UCB.
        self._acquisition_function = ucb


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # self.likelihood_f = gpytorch.likelihoods.GaussianLikelihood()
        # self.likelihood_v = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood_f = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.FloatTensor([0.15]))
        self.likelihood_v = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.FloatTensor([0.0001]))

        self.no_data_yet = True

        self.X = []
        self.f, self.v = [], []

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
        return self.optimize_acquisition_function()


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

        return np.array(x_opt).reshape(-1, 1)

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
        # self.algorithm_f.update_acquisition_function(x)
        # self.algorithm_v.update_acquisition_function(x)

        """
        beta = 2.0
        gamma = 1.5

        pred_f = self.algorithm_f.gp(torch.FloatTensor(x))
        pred_v = self.algorithm_v.gp(torch.FloatTensor(x))
        mean, std = pred_f.mean.detach().numpy(), pred_f.stddev.detach().numpy()
        mean_v, std_v = pred_v.mean.detach().numpy(), pred_v.stddev.detach().numpy()

        # return mean + beta * std - gamma * np.max([float(mean_v), 0.])
        return mean + beta * std - 1e4 * np.max([float(mean_v + 1.0 * std_v)-3.5, 0.])
        """
        pred_f = self.algorithm_f.gp(torch.FloatTensor(x))
        pred_v = self.algorithm_v.gp(torch.FloatTensor(x))
        f_sample = pred_f.sample()
        v_sample = pred_v.sample()

        return f_sample - 1e4 * np.max([float(v_sample)-3.5, 0.])

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

        self.X.append(float(x))
        self.f.append(float(f))
        self.v.append(float(v))

        if self.no_data_yet:
            x, f, v = [torch.FloatTensor([i]) for i in [x, f, v]]  # convert to Tensor
            self.no_data_yet = False
            gp_f = GP(x, f, self.likelihood_f, type='f')
            gp_v = GP(x, v, self.likelihood_v, type='v')

            self.algorithm_f = GPUCB(gp_f, x)
            self.algorithm_v = GPUCB(gp_v, x)

        else:
            self.algorithm_f.update_gp(np.array([x]), np.array([f]))
            self.algorithm_v.update_gp(np.array([x]), np.array([float(v)]))

        # print(f"length scale of f is {self.algorithm_f.gp.length_scale}")
        # print(f"length scale of v is {self.algorithm_v.gp.length_scale}")


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        def objective(x):
            x_tens = torch.FloatTensor(x)
            v_pred = self.algorithm_v.gp(x_tens)
            if v_pred.mean.detach().numpy() + 5 * v_pred.stddev.detach().numpy() > 3.9:
                return 1e5 * (v_pred.mean.detach().numpy() + 5 * v_pred.stddev.detach().numpy() - 3.9)
            return -self.algorithm_f.gp(x_tens).mean.detach().numpy()

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

        return np.array(x_opt).reshape(-1, 1)

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.

        """

        x = np.linspace(0, 10, num=1000)
        plt.figure(figsize=(10, 6))
        ff, vv = np.array([f(i) for i in x]), np.array([v(i) for i in x])
        plt.plot(x, ff, 'k--', label="True F Function")
        plt.plot(x, vv, 'k--', label="True v Function")
        plt.scatter(self.X, self.f, color='blue', label='F')
        plt.scatter(self.X, self.v, color='red', label='v')

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.algorithm_f.gp(torch.FloatTensor(x))
            mean = pred.mean.detach().numpy()
            error = 2 * pred.stddev.detach().numpy()
            pred_v = self.algorithm_v.gp(torch.FloatTensor(x))
            mean_v = pred_v.mean.detach().numpy()
            error_v = 2 * pred_v.stddev.detach().numpy()

        plt.fill_between(x, mean - error, mean + error, lw=0, alpha=0.2, color='C0')        # out = self.algorithm_f.learner.predict_normal(x)
        plt.fill_between(x, mean_v - error_v, mean_v + error_v, lw=0, alpha=0.2, color='salmon')
        plt.plot(x, mean, lw=2, color='C0')
        plt.plot(x, mean_v, lw=2, color='salmon')
        plt.legend()
        plt.show()

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
    return 3.0 + 2 * np.sin(x+0.1)


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
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

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.normal(loc=0, scale=0.1)
        cost_val = v(x) + np.random.normal(loc=0, scale=0.001)
        agent.add_data_point(x, obj_val, cost_val)
        agent.plot()

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

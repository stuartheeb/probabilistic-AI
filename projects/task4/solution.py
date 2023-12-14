import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings
from typing import Union
from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.constant_(m.weight, 0.)
        torch.nn.init.constant_(m.bias, 0.)


class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                 hidden_layers: int, activation: str = "relu"):
        super(NeuralNetwork, self).__init__()
        layers = []

        # Add input layer to the network
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())

        # Add hidden layers to the network
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh() if activation == 'tanh' else nn.ReLU())

        # Add output layer to the network
        layers.append(nn.Linear(hidden_size, output_dim))

        # init weights
        self.apply(weights_init_)

        # Define the neural network using Sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.model(s)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = nn.functional.relu(self.linear1(state))
        x = nn.functional.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # std = torch.clip(std, -0.4, 0.4)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        # log_prob = log_prob.mean(0, keepdim=True)
        # log_prob = log_prob.mean(0, keepdim=True)         # mean and not sum!!! (or ignore)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = nn.functional.relu(self.linear1(state))
        x = nn.functional.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.4)     # standard is 0.1
        noise = noise.clamp(-0.3, 0.3)          # standard is 0.25
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


class Actor:
    def __init__(self, hidden_size: int, hidden_layers: int, actor_lr: float,
                 state_dim: int = 3, action_dim: int = 1, policy_type='deterministic', automatic_entropy_tuning=True,
                 device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.model, self.optimizer = [None] * 2
        self.policy_type = policy_type
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        if self.policy_type == 'gaussian':
            print("using gaussian policy")
            self.model = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size).to(self.device)

            if self.automatic_entropy_tuning:
                self.target_entropy = -1.  # - dimension of action space
                self.log_alpha = torch.tensor([np.log(0.2)], requires_grad=True, dtype=torch.float32, device=self.device)
                self.alpha_optim = optim.Adam([self.log_alpha], lr=self.actor_lr)
        else:
            print(f"using {self.policy_type} policy")
            self.model = DeterministicPolicy(self.state_dim, self.action_dim, self.hidden_size).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action 
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # Implement this function which returns an action and its log probability.

        action, log_prob, mean = self.model.sample(state=state)
        action = mean if deterministic else action

        return action, log_prob

    def alpha_update_step(self, log_pi, alpha):
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            return self.log_alpha.exp()
        else:
            return alpha


class Critic:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, critic_lr: float, state_dim: int = 3,
                 action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.Q1, self.Q2, self.optimizer = [None] * 3
        self.setup_critic()

    def setup_critic(self):
        # Implement this function which sets up the critic(s).
        self.Q1 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size,
                                self.hidden_layers).to(self.device)
        self.Q2 = NeuralNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size,
                                self.hidden_layers).to(self.device)
        self.optimizer = optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=self.critic_lr)

    def __call__(self, state, action):
        both = torch.cat((state, action), dim=1)
        return self.Q1(both), self.Q2(both)


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 1000
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        self.updates_per_step = 1
        self.alpha = 0.2  # 0.2
        self.gamma = 0.95  # 0.99
        self.tau = 0.005  # 0.005
        self.lr = 0.001
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        self.policy, self.critic, self.critic_target = [None] * 3
        self.setup_agent()

    def setup_agent(self):
        # Setup off-policy agent with policy and critic classes.
        # Feel free to instantiate any other parameters you feel you might need. 
        self.policy = Actor(64, 2, self.lr, self.state_dim, self.action_dim,
                            policy_type='gaussian', automatic_entropy_tuning=True, device=self.device)
        self.critic = Critic(64, 2, self.lr, self.state_dim, self.action_dim, self.device)
        self.critic_target = Critic(64, 2, self.lr, self.state_dim, self.action_dim, self.device)

        # hard copy weights to target
        self.critic_target_update(self.critic.Q1, self.critic_target.Q1, self.tau, False)
        self.critic_target_update(self.critic.Q2, self.critic_target.Q2, self.tau, False)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray, action to apply on the environment, shape (1,)
        """
        # Implement a function that returns an action from the policy for the state s.

        DETERMNIISTIC = not train
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        action, _ = self.policy.get_action_and_log_prob(state=s, deterministic=DETERMNIISTIC)
        action = np.clip(action.cpu().detach().numpy(), -1., 1.)

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray), 'Action dtype must be np.ndarray'

        # if state is between [90, 270]: hard set to +-1 (quicker conversion)
        if not train and s[0] < 0.:
            action = np.array(1.) * np.sign(action)

        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer, 
        and using a given loss, runs one step of gradient update. If you set up trainable parameters 
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent_iteration(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample(self.batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.model.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = nn.functional.mse_loss(qf1, next_q_value)
        qf2_loss = nn.functional.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.run_gradient_update_step(self.critic, qf_loss)

        pi, log_pi, _ = self.policy.model.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.run_gradient_update_step(self.policy, policy_loss)

        # update alpha if automatic_entropy_tuning is true, else leave it
        self.alpha = self.policy.alpha_update_step(log_pi=log_pi, alpha=self.alpha)

        # soft update
        self.critic_target_update(self.critic.Q1, self.critic_target.Q1, self.tau, True)
        self.critic_target_update(self.critic.Q2, self.critic_target.Q2, self.tau, True)
        # soft_copy(self.critic_target, self.critic, self.tau)
        # return qf1_loss.item(), qf2_loss.item(), policy_loss.item()

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # Implement one step of training for the agent.

        # multiple iterations per step
        for i in range(self.updates_per_step):
            self.train_agent_iteration()


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':
    TRAIN_EPISODES = 80  # 50
    TEST_EPISODES = 10  # 300

    # You may set the save_video param to output the video of one of the evalution episodes, or 
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()

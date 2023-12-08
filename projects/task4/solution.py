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

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(QNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        layers = []

        # Add input layer to the network
        layers.append(nn.Linear(input_dim, hidden_size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Add hidden layers to the network
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

        # Add output layer to the network
        layers.append(nn.Linear(hidden_size, output_dim))

        # Define the neural network using Sequential container
        self.model = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        return self.model(s)
    
class PolicyNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int, 
                                hidden_layers: int, activation: str):
        super(PolicyNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network 
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        layers = []

        # Add input layer to the network
        layers.append(nn.Linear(input_dim, hidden_size))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # Add hidden layers to the network
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

        # Define the neural network using Sequential container
        self.model = nn.Sequential(*layers)
        self.mean = nn.Linear(hidden_size, output_dim)
        self.log_std = nn.Linear(hidden_size, output_dim)
        self.apply(weights_init)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        x = self.model(s)
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std
    
class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        self.model = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_size, self.hidden_layers, 'relu').to(self.device)
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
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped 
        # using the clamp_log_std function.
        action, log_prob = self.model(state)
        log_prob = self.clamp_log_std(log_prob)
        if(deterministic):
            action = torch.tanh(action)
        else:
            std = log_prob.exp()
            normal = Normal(action, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 10e-6)
            log_prob = log_prob.sum(1, keepdim=True)


        # assert action.shape == (state.shape[0], self.action_dim) and \
        #     log_prob.shape == (state.shape[0], self.action_dim), 'Incorrect shape for action or log_prob.'

        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int, 
                 hidden_layers: int, critic_lr: int, state_dim: int = 3, 
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork 
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        self.Q1 = QNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers, 'relu').to(self.device)
        self.Q2 = QNetwork(self.state_dim + self.action_dim, self.action_dim, self.hidden_size, self.hidden_layers, 'relu').to(self.device)
        self.optimizer = optim.Adam(list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=self.critic_lr)


class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it 
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float, 
                 train_param: bool, device: torch.device = torch.device('cpu')):
        
        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training, 
        # as self.device should be 'cuda' in that case.
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available(): 
            self.device = torch.device("cuda") 
        elif torch.backends.mps.is_available(): 
            self.device = torch.device("mps") 
        else: 
            self.device = torch.device("cpu") 
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
        
        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes. 
        # Feel free to instantiate any other parameters you feel you might need. 
        self.actor = Actor(128, 3, 0.001, self.state_dim, self.action_dim, self.device)
        self.critic = Critic(128, 3, 0.001, self.state_dim, self.action_dim, self.device)
        self.critic_target_update(self.critic.Q1, self.critic.Q2, None, False)
        

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode. 
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        DETERMNIISTIC = True
        s = torch.tensor(s, dtype=torch.float, device=self.device)
        if train:
            action, _ = self.actor.get_action_and_log_prob(s, DETERMNIISTIC)
        else:
            action, _ = self.actor.model(s)
        action = action.cpu().detach().numpy()
        action = action.clip(-1, 1)
        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray' 
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

    def critic_target_update(self, base_net: QNetwork, target_net: QNetwork, 
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

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch 
        from the replay buffer,and then updates the policy and critic networks 
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch

        deterministic = False
        alpha = 0.4
        gamma = 0.9
        tau = 0.01
        # TODO: Implement Critic(s) update here.
        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.get_action_and_log_prob(s_prime_batch, deterministic)
            q1_next_target = self.critic.Q1(torch.cat((s_prime_batch, next_state_action),dim=1))
            q2_next_target = self.critic.Q2(torch.cat((s_prime_batch, next_state_action),dim=1))
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - alpha * next_state_log_pi
            next_q_value = r_batch + gamma * min_q_next_target
        q1 = self.critic.Q1(torch.cat((s_batch, a_batch), dim=1))
        q2 = self.critic.Q2(torch.cat((s_batch, a_batch), dim=1))
        q1_loss = nn.MSELoss()(q1, next_q_value)
        q2_loss = nn.MSELoss()(q2, next_q_value)
        q_loss = q1_loss + q2_loss

        self.run_gradient_update_step(self.critic, q_loss)

        pi, log_pi = self.actor.get_action_and_log_prob(s_batch, deterministic)
        q1_pi = self.critic.Q1(torch.cat((s_batch, pi), dim=1))
        q2_pi = self.critic.Q2(torch.cat((s_batch, pi), dim=1))
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((alpha * log_pi) - min_q_pi).mean()

        self.run_gradient_update_step(self.actor, policy_loss)

        self.critic_target_update(self.critic.Q1, self.critic.Q2, tau, True)




# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 30
    TEST_EPISODES = 300

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

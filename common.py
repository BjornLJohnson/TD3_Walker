import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def calculate_return(rollout, gamma):
    r = 0
    for transition in rollout:
        r = gamma*r + transition['reward']

    return r


def update_target_model(target_model, source_model, tau=0.001):
    with torch.no_grad():
        update_params = deepcopy(target_model.state_dict())
        for name, value in source_model.named_parameters():
            update_params[name] = tau*value + (1-tau)*update_params[name]
        target_model.load_state_dict(update_params)


def visualize_rollout(env, actor):
    # Evaluate the final policy
    state = env.reset()
    done = False
    while not done:
        s_tens = torch.tensor(state, dtype=torch.float)
        action = actor(s_tens).detach().squeeze().numpy()
        next_state, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)
        state = next_state


def collect_episode(env, actor=None, add_noise=False, random=False):
    action_dim = env.action_space._shape[0]
    m = torch.distributions.MultivariateNormal(torch.zeros((action_dim,)), torch.eye(action_dim)*0.1)
    transitions = []
    s = env.reset()
    done = False
    while not done:
        if random:
            a = env.action_space.sample()
        else:
            s_tens = torch.tensor(s, dtype=torch.float)
            a_tens = actor(s_tens)
            if add_noise:
                a_tens += m.sample()
            torch.clamp(a_tens, min=-1, max=1)
            a = a_tens.detach().numpy()
        s_prime, r, done, _ = env.step(a)
        transitions.append({
            'state': s,
            'action': a,
            'reward': r,
            'next_state': s_prime
        })
        s = s_prime
    return transitions


def fill_transition_buffer(env, buffer, num_steps, actor=None, add_noise=False, random=False):
    steps = 0

    while steps < num_steps:
        transitions = collect_episode(env, actor, add_noise, random)
        steps += len(transitions)
        buffer.buffer_add(transitions)

    return steps


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        """
        A function to initialize the replay buffer.

        param: init_length : Initial number of transitions to collect
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        param: env : gym environment object
        """

        total_dim = 2*state_dim+action_dim+1

        self.state_slice = slice(0, state_dim)
        self.action_slice = slice(state_dim, state_dim+action_dim)
        self.reward_ind = state_dim + action_dim
        self.next_state_slice = slice(state_dim+action_dim+1, total_dim)

        self.max_len = buffer_size
        self.memory = torch.zeros((buffer_size, total_dim))
        self.curr_ind = 0
        self.stored_transitions = 0

    def buffer_add(self, transitions):
        """
        A function to add a dictionary to the buffer
        param: exp : A dictionary consisting of state, action, reward , next state and done flag
        """
        for t in transitions:
            s = torch.tensor(t['state'], dtype=torch.float)
            a = torch.tensor(t['action'], dtype=torch.float)
            r = torch.tensor(t['reward'], dtype=torch.float).unsqueeze(dim=0)
            n = torch.tensor(t['next_state'], dtype=torch.float)

            self.add_one_transition(s, a, r, n)

    def add_one_transition(self, s, a, r, n):
        self.memory[self.curr_ind] = torch.cat([s, a, r, n])
        self.curr_ind = (self.curr_ind + 1) % self.max_len
        if self.stored_transitions < self.max_len:
            self.stored_transitions += 1

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer
        param: N : Number of samples to obtain from the buffer
        """
        indices = torch.randperm(self.stored_transitions)[:N]

        samples = self.memory[indices]

        states = samples[:, self.state_slice]
        actions = samples[:, self.action_slice]
        rewards = samples[:, self.reward_ind].unsqueeze(dim=1)
        next_states = samples[:, self.next_state_slice]

        return states, actions, rewards, next_states


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the network
        param: state_dim : Size of the state space
        param: action_dim: Size of the action space
        """
        super(Actor, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, action_dim)

    def forward(self, x):
        """
        Define the forward pass
        param: state: The state of the environment
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic
        param: state_dim : Size of the state space
        param: action_dim : Size of the action space
        """
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, state, action):
        """
        Define the forward pass of the critic
        """
        x = torch.cat([state, action], axis=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

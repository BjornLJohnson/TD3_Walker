""" Learn a policy using DDPG for the reach task"""
import gym
import matplotlib.pyplot as plt
import torch

from common import *


class TD3:
    def __init__(
            self,
            env,
            critic_lr,
            actor_lr,
            gamma,
            batch_size_sample,
            d,
            batch_size_generate,
            tau,
            action_noise_cov,
            a_tilde_cov
    ):
        """
        param: env: An gym environment
        param: action_dim: Size of action space
        param: state_dim: Size of state space
        param: critic_lr: Learning rate of the critic
        param: actor_lr: Learning rate of the actor
        param: gamma: The discount factor
        param: batch_size: The batch size for training
        """
        self.env = env
        self.gamma = gamma # discount factor
        self.batch_size_sample = batch_size_sample # number of transitions to sample at once
        self.batch_size_generate = batch_size_generate  # number of transitions to generate at once
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.d = d  # number of critic steps before stepping actor/targets
        self.tau = tau  # how far to step target net toward trained nets

        self.action_noise_cov = torch.eye(self.action_dim) * action_noise_cov
        self.a_tilde_cov = torch.eye(self.action_dim) * a_tilde_cov

        self.actor = Actor(self.state_dim, self.action_dim, self.env.action_space.low, self.env.action_space.high)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.env.action_space.low, self.env.action_space.high)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(self.state_dim, self.action_dim)
        self.critic_1_target = Critic(self.state_dim, self.action_dim)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())

        self.critic_2 = Critic(self.state_dim, self.action_dim)
        self.critic_2_target = Critic(self.state_dim, self.action_dim)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.ReplayBuffer = ReplayBuffer(10000, self.state_dim, self.action_dim)
        fill_transition_buffer(self.env, self.ReplayBuffer, 1000, random=True)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        update_target_model(self.actor_target, self.actor, self.tau)
        update_target_model(self.critic_1_target, self.critic_1, self.tau)
        update_target_model(self.critic_2_target, self.critic_2, self.tau)

    def update_actor(self, s):
        on_policy_a = self.actor(s)
        on_policy_Q1 = self.critic_1(s, on_policy_a)
        on_policy_Q2 = self.critic_2(s, on_policy_a)

        # TODO: Make sure using min for policy updates is consistent w/ TD3
        min_Q = torch.min(on_policy_Q1, on_policy_Q2)

        assert on_policy_a.shape == (s.shape[0], self.action_dim)
        assert min_Q.shape == (s.shape[0], 1)

        actor_loss = -torch.mean(min_Q)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.detach()

    def update_critics(self, s, a, r, s_prime):

        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(a.shape), self.a_tilde_cov)
        epsilon = noise_dist.sample()
        # TODO: find a suitable cov value and clip noise to be in a defined range
        # epsilon = torch.clip(epsilon, -c, c)

        assert epsilon.shape == a.shape

        with torch.no_grad():
            a_target = self.actor_target(s_prime)

            assert a_target.shape == a.shape

            a_tilde = a_target + epsilon
            a_tilde = torch.clip(a_tilde, -1, 1)

            assert a_tilde.shape == a.shape

            Q1_tilde = self.critic_1_target(s_prime, a_tilde)
            Q2_tilde = self.critic_2_target(s_prime, a_tilde)

            assert Q1_tilde.shape == r.shape

            # TODO: use y = r where s_prime is terminal as a result of a failure state?
            y = r + self.gamma * torch.min(Q1_tilde, Q2_tilde)

        Q1 = self.critic_1(s, a)
        Q2 = self.critic_2(s, a)

        assert Q1.shape == y.shape

        criterion = torch.nn.MSELoss()
        critic_loss_1 = criterion(y, Q1)
        critic_loss_2 = criterion(y, Q2)

        self.optimizer_critic_1.zero_grad()
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_loss_2.backward()
        self.optimizer_critic_2.step()

        return critic_loss_1.detach()

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        critic_losses = []
        actor_losses = []
        returns = []
        for t in range(num_steps):
            # TODO: Replace this with a single transition instead of an entire episode
            fill_transition_buffer(self.env, self.ReplayBuffer, self.batch_size_generate, self.actor, noise_cov=self.action_noise_cov)

            s, a, r, s_prime = self.ReplayBuffer.buffer_sample(self.batch_size_sample)

            assert s.shape[1] == self.state_dim
            assert a.shape[1] == self.action_dim
            assert r.shape[1] == 1
            assert s_prime.shape[1] == self.state_dim

            critic_loss = self.update_critics(s, a, r, s_prime)

            if t % self.d == 0:
                actor_loss = self.update_actor(s)
                self.update_target_networks()

                rollout = collect_episode(self.env, self.actor, noise_cov=self.action_noise_cov)
                epi_return = calculate_return(rollout, self.gamma)

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
                returns.append(epi_return)

                print("~~~~~~~~~~~~~~~~~~~")
                print("Train Step: {}".format(t))
                print("Critic loss: {}".format(critic_loss))
                print("Actor loss: {}".format(actor_loss))
                print("Eval epi return: {}".format(epi_return))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(actor_losses)
        ax1.title.set_text("Actor Loss")
        ax2.plot(critic_losses)
        ax2.title.set_text("Critic Loss")
        ax3.plot(returns)
        ax3.title.set_text("Returns")
        plt.show()


if __name__ == "__main__":
    # Define the environment
    # env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    # env = gym.make("Walker2d-v2")
    # env = gym.make("InvertedPendulum-v2")
    env = gym.make("Pendulum-v1")
    # env = gym.make("CartPole-v1")

    # TODO: Sync hyper parameters and network architectures with paper
    td3_object = TD3(
        env,
        critic_lr=1e-3,
        actor_lr=1e-3,
        gamma=0.99,
        batch_size_sample=100,
        batch_size_generate=10,
        d=2,  # number of critic updates per actor update
        tau=5e-3,  # how far to step targets towards trained policies
        action_noise_cov=0.01,  # noise to add to policy output in rollouts
        a_tilde_cov=1e-4  # noise to add to actions for q function estimates
    )

    # visualize_rollout(env, td3_object.actor)

    # Train the policy
    td3_object.train(int(1e4))

    visualize_rollout(env, td3_object.actor)


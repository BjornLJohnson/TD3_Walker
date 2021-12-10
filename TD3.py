""" Learn a policy using DDPG for the reach task"""
import gym
import matplotlib.pyplot as plt
import torch

from pyinstrument import Profiler
from common import ReplayBuffer, Actor, Critic, update_target_model, visualize_rollout, collect_episode, calculate_return


class TD3:
    def __init__(
            self,
            env,
            critic_lr,
            actor_lr,
            gamma,
            batch_size_sample,
            policy_update_freq,
            batch_size_generate,
            tau,
            exploration_noise,
            a_tilde_noise
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
        self.gamma = gamma  # discount factor
        self.batch_size_sample = batch_size_sample  # number of transitions to sample at once
        self.batch_size_generate = batch_size_generate  # number of transitions to generate at once
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.policy_update_freq = policy_update_freq  # number of critic steps before stepping actor/targets
        self.tau = tau  # how far to step target net toward trained nets

        self.action_noise_cov = torch.eye(self.action_dim) * exploration_noise

        self.a_tilde_noise = a_tilde_noise
        self.a_tilde_noise_clip = 0.5

        self.env_done = False
        self.env_s = self.env.reset()
        self.action_noise_dist = torch.distributions.MultivariateNormal(torch.zeros((self.action_dim,)), self.action_noise_cov)
        self.action_min = torch.tensor(env.action_space.low)
        self.action_max = torch.tensor(env.action_space.high)

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
        self.store_transitions(1000, randomize_action=True)

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
        # on_policy_Q2 = self.critic_2(s, on_policy_a)

        # TODO: Make sure using min for policy updates is consistent w/ TD3
        # min_Q = torch.min(on_policy_Q1, on_policy_Q2)

        assert on_policy_a.shape == (s.shape[0], self.action_dim)
        # assert min_Q.shape == (s.shape[0], 1)

        # actor_loss = -torch.mean(min_Q)
        actor_loss = -torch.mean(on_policy_Q1)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.detach()

    def store_transitions(self, num_steps, randomize_action):
        for _ in range(num_steps):
            self.store_single_transition(randomize_action)

    def store_single_transition(self, randomize_action):
        with torch.no_grad():
            if self.env_done:
                self.env_s = env.reset()

            if randomize_action:
                a = env.action_space.sample()
            else:
                s_tens = torch.tensor(self.env_s, dtype=torch.float)
                a_tens = self.actor(s_tens)
                a_tens += self.action_noise_dist.sample()
                torch.clamp(a_tens, min=self.action_min, max=self.action_max)
                a = a_tens.detach().numpy()
            s_prime, r, self.env_done, _ = env.step(a)
            self.ReplayBuffer.add_one(self.env_s, a, r, s_prime, not self.env_done)
            self.env_s = s_prime

    def update_critics(self, s, a, r, s_prime, not_done):
        with torch.no_grad():
            epsilon = torch.randn_like(a) * self.a_tilde_noise
            epsilon = torch.clip(epsilon, -self.a_tilde_noise_clip, self.a_tilde_noise_clip)

            a_tilde = self.actor_target(s_prime) + epsilon
            a_tilde = torch.clip(a_tilde, self.action_min, self.action_max)

            assert a_tilde.shape == a.shape

            Q1_tilde = self.critic_1_target(s_prime, a_tilde)
            Q2_tilde = self.critic_2_target(s_prime, a_tilde)

            assert Q1_tilde.shape == r.shape

            y = r + not_done * self.gamma * torch.min(Q1_tilde, Q2_tilde)

        Q1 = self.critic_1(s, a)
        Q2 = self.critic_2(s, a)

        assert Q1.shape == y.shape

        critic_loss_1 = torch.nn.functional.mse_loss(y, Q1)
        critic_loss_2 = torch.nn.functional.mse_loss(y, Q2)

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
            self.store_transitions(self.batch_size_generate, randomize_action=False)

            s_batch, a_batch, r_batch, s_prime_batch, not_done_batch = self.ReplayBuffer.sample(self.batch_size_sample)

            assert s_batch.shape[1] == self.state_dim
            assert a_batch.shape[1] == self.action_dim
            assert r_batch.shape[1] == 1
            assert s_prime_batch.shape[1] == self.state_dim
            assert not_done_batch.shape[1] == 1

            critic_loss = self.update_critics(s_batch, a_batch, r_batch, s_prime_batch, not_done_batch)

            if t % self.policy_update_freq == 0:
                actor_loss = self.update_actor(s_batch)
                self.update_target_networks()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)

            if t % 500 == 0:
                rollout = collect_episode(self.env, self.actor, noise_cov=self.action_noise_cov)
                epi_return = calculate_return(rollout, self.gamma)

                returns.append(epi_return)

                print("~~~~~~~~~~~~~~~~~~~")
                print("Epoch: {}".format(t))
                print("Generated Transitions: {}".format(t*self.batch_size_generate))
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
    # env = gym.make("Reacher-v2")
    # env = gym.make("Walker2d-v2")
    # env = gym.make("InvertedPendulum-v2")
    env = gym.make("Pendulum-v1")
    # env = gym.make("CartPole-v1")

    td3_object = TD3(
        env,
        critic_lr=3e-4,
        actor_lr=3e-4,
        gamma=0.99,
        batch_size_sample=256,
        batch_size_generate=1,
        policy_update_freq=5,  # number of critic updates per actor update
        tau=5e-3,  # how far to step targets towards trained policies
        exploration_noise=0.1,  # noise to add to policy output in rollouts
        a_tilde_noise=0.2  # noise to add to actions for q function estimates
    )

    # profiler = Profiler()
    # profiler.start()

    # Train the policy
    td3_object.train(int(2e4))

    # profiler.stop()
    # profiler.print()

    visualize_rollout(env, td3_object.actor)


""" Learn a policy using DDPG for the reach task"""
import time
import gym
import matplotlib.pyplot as plt
from common import *


class TD3:
    def __init__(
            self,
            env,
            state_dim,
            action_dim,
            critic_lr,
            actor_lr,
            gamma,
            batch_size_sample,
            d,
            batch_size_generate
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
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.d = d  # number of critic steps before stepping actor/targets

        self.a_tilde_cov = torch.eye(action_dim) * 0.01

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_1_target.load_state_dict(self.critic.state_dict())

        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        self.critic_2_target.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.optimizer_critic_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.ReplayBuffer = ReplayBuffer(10000, state_dim, action_dim)
        fill_transition_buffer(self.env, self.ReplayBuffer, 1000, random=True)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        update_target_model(self.actor_target, self.actor)
        update_target_model(self.critic_1_target, self.critic_1)
        update_target_model(self.critic_2_target, self.critic_2)

    def update_actor(self, s):
        on_policy_a = self.actor(s)
        on_policy_Q = self.critic_1(s, on_policy_a)

        self.optimizer_actor.zero_grad()
        actor_loss = torch.mean(on_policy_Q)
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.detach()

    def update_critics(self, s, a, r, s_prime):

        noise_dist = torch.distributions.MultivariateNormal(torch.zeros(a.shape), self.a_tilde_cov)
        epsilon = noise_dist.sample()

        a_tilde = self.actor_target(s_prime) + epsilon
        Q1_tilde = self.critic_1_target(s_prime, a_tilde)
        Q2_tilde = self.critic_2_target(s_prime, a_tilde)
        y = r + self.gamma * torch.min(Q1_tilde, Q2_tilde)

        Q1 = self.critic_1(s, a)
        Q2 = self.critic_2(s, a)

        criterion = torch.MeanSquaredError()

        self.optimizer_critic_1.zero_grad()
        critic_loss_1 = criterion(y, Q1)
        critic_loss_1.backward()
        self.optimizer_critic_1.step()

        self.optimizer_critic_2.zero_grad()
        critic_loss_2 = criterion(y, Q2)
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
            fill_transition_buffer(self.env, self.ReplayBuffer, self.actor, self.batch_size_generate, add_noise=True)

            s, a, r, s_prime = self.ReplayBuffer.buffer_sample(self.batch_size_sample)

            critic_loss = self.update_critics(s, a, r, s_prime)

            if t % self.d == 0:
                actor_loss = self.update_actor(s)
                self.update_target_networks()

                rollout = collect_episode(self.env, self.actor, add_noise=False)
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
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    env.render()

    td3_object = TD3(
        env,
        state_dim=8,
        action_dim=2,
        critic_lr=1e-4,
        actor_lr=1e-4,
        gamma=0.99,
        batch_size_sample=100,
        batch_size_generate=10,
        d=10
    )

    # Train the policy
    td3_object.train(1e4)

    visualize_rollout(env, td3_object.actor)

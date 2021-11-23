""" Learn a policy using DDPG for the reach task"""
import time
import gym
import matplotlib.pyplot as plt
from common import *


class DDPG:
    def __init__(
            self,
            env,
            state_dim,
            action_dim,
            critic_lr,
            actor_lr,
            gamma,
            batch_size,
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
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.ReplayBuffer = ReplayBuffer(10000, self.state_dim, self.action_dim)
        fill_transition_buffer(self.env, self.ReplayBuffer, 1000, random=True)

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        update_target_model(self.actor_target, self.actor)
        update_target_model(self.critic_target, self.critic)

    def update_critic(self, state_batch, action_batch, next_state_batch, reward_batch):
        self.optimizer_critic.zero_grad()

        next_action_batch = self.actor_target(next_state_batch)
        next_state_Q = self.critic_target(next_state_batch, next_action_batch)
        expected_Q = reward_batch + self.gamma * next_state_Q

        Q_sampled_actions = self.critic(state_batch, action_batch)

        criterion = torch.nn.MSELoss()
        # print(expected_Q.shape)
        # print(Q_sampled_actions.shape)
        critic_loss = criterion(expected_Q, Q_sampled_actions)
        critic_loss.backward()

        # for param in self.critic.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer_critic.step()

        return critic_loss.detach()

    def update_actor(self, state_batch):
        self.optimizer_actor.zero_grad()

        policy_action_batch = self.actor(state_batch)
        Q_policy = self.critic(state_batch, policy_action_batch)

        actor_loss = -torch.mean(Q_policy)
        actor_loss.backward()

        # for param in self.actor.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer_actor.step()

        return actor_loss.detach()

    def update_network(self):
        """
        A function to update the function just once
        """
        state_batch, action_batch, reward_batch, next_state_batch = self.ReplayBuffer.buffer_sample(self.batch_size)

        critic_loss = self.update_critic(state_batch, action_batch, next_state_batch, reward_batch)

        actor_loss = self.update_actor(state_batch)

        return actor_loss, critic_loss

    def train(self, num_steps, target_update_interval, num_new_transitions_per_epoch):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        :param target_update_interval:The number of times to update the policy before the target
        """
        critic_losses = []
        actor_losses = []
        returns = []
        steps = 0
        for epoch in range(num_steps):
            steps += fill_transition_buffer(self.env, self.ReplayBuffer, num_new_transitions_per_epoch, self.actor, add_noise=True)
            for _ in range(target_update_interval):
                actor_loss, critic_loss = self.update_network()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)

            if epoch % 10 == 0:
                print("~~~~~~~~~~~~~~~~~~~")
                print("Epoch: {}".format(epoch))
                print("Total steps: {}".format(steps))
                print("Critic loss: {}".format(critic_loss))
                print("Actor loss: {}".format(actor_loss))

            self.update_target_networks()

            rollout = collect_episode(self.env, self.actor, self.action_dim, add_noise=False)
            returns.append(calculate_return(rollout, self.gamma))

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

    ddpg_object = DDPG(
        env,
        state_dim=8,
        action_dim=2,
        critic_lr=1e-4,
        actor_lr=1e-4,
        gamma=0.99,
        batch_size=100,
    )

    # Train the policy
    ddpg_object.train(200, 10, 1000)

    visualize_rollout(env, ddpg_object.actor)

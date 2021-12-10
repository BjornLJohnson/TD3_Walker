import gym
import matplotlib.pyplot as plt
import torch

from pyinstrument import Profiler
from common import ReplayBuffer, Actor, Critic, update_target_model


class DDPG:
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

        self.env_done = False
        self.env_s = self.env.reset()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_grad_norm = 5

        self.a_tilde_noise = a_tilde_noise
        self.a_tilde_noise_clip = 0.5

        self.action_noise_cov = torch.eye(self.action_dim) * exploration_noise
        self.action_noise_dist = torch.distributions.MultivariateNormal(torch.zeros((self.action_dim,)), self.action_noise_cov)
        self.action_min = torch.tensor(env.action_space.low).to(self.device)
        self.action_max = torch.tensor(env.action_space.high).to(self.device)

        self.actor = Actor(self.state_dim, self.action_dim, self.env.action_space.low, self.env.action_space.high).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.env.action_space.low, self.env.action_space.high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.ReplayBuffer = ReplayBuffer(int(1e6), self.state_dim, self.action_dim)
        self.store_transitions(int(25e3), randomize_action=True)

        self.logs = {
            "critic_loss": [],
            "actor_loss": [],
            "eval_epi_return": [],
            "critic_grad_norm": [],
            "actor_grad_norm": []
        }

    def log(self, value, key):
        self.logs[key].append(value)

    def make_graphs(self):
        fig, axs = plt.subplots(len(self.logs), 1)
        i = 0
        # print(x for x in self.logs)
        for key, values in self.logs.items():
            axs[i].plot(values)
            axs[i].title.set_text(key)
            i += 1

        plt.show()

    def print_update(self, t):

        print("~~~~~~~~~~~~~~~~~~~")
        print("Epoch: {}".format(t))
        print("Generated Transitions: {}".format(t*self.batch_size_generate))
        print("Critic loss: {}".format(self.logs["critic_loss"][-1]))
        print("Actor loss: {}".format(self.logs["actor_loss"][-1]))
        print("Eval epi return: {}".format(self.logs["eval_epi_return"][-1]))

    def update_target_networks(self):
        """
        A function to update the target networks
        """
        update_target_model(self.actor_target, self.actor, self.tau)
        update_target_model(self.critic_target, self.critic, self.tau)

    def update_actor(self, s):
        on_policy_a = self.actor(s)
        on_policy_Q = self.critic(s, on_policy_a)

        assert on_policy_a.shape == (s.shape[0], self.action_dim)

        actor_loss = -torch.mean(on_policy_Q)
        self.log(actor_loss.data, "actor_loss")

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm, norm_type=2.0)
        self.log(grad_norm.data, "actor_grad_norm")
        self.optimizer_actor.step()

    def update_critic(self, s, a, r, s_prime, not_done):
        with torch.no_grad():
            epsilon = torch.randn_like(a) * self.a_tilde_noise
            epsilon = torch.clip(epsilon, -self.a_tilde_noise_clip, self.a_tilde_noise_clip)

            a_tilde = self.actor_target(s_prime) + epsilon
            a_tilde = torch.clip(a_tilde, self.action_min, self.action_max)

            assert a_tilde.shape == a.shape

            Q_tilde = self.critic_target(s_prime, a_tilde)

            assert Q_tilde.shape == r.shape

            y = r + not_done * self.gamma * Q_tilde

        Q = self.critic(s, a)

        assert Q.shape == y.shape

        critic_loss = torch.nn.functional.mse_loss(y, Q)
        self.log(critic_loss.data, "critic_loss")

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm, norm_type=2.0)
        self.log(grad_norm.data, "critic_grad_norm")
        self.optimizer_critic.step()

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
                s_tens = torch.tensor(self.env_s, dtype=torch.float).to(self.device)
                a_tens = self.actor(s_tens)
                a_tens += self.action_noise_dist.sample().to(self.device)
                torch.clamp(a_tens, min=self.action_min, max=self.action_max)
                a = a_tens.detach().numpy()
            s_prime, r, self.env_done, _ = env.step(a)
            self.ReplayBuffer.add_one(self.env_s, a, r, s_prime, not self.env_done)
            self.env_s = s_prime

    def visualize_rollout(self):
        # Evaluate the final policy
        state = env.reset()
        done = False
        epi_return = 0
        while not done:
            s_tens = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.actor(s_tens).detach().numpy()
            next_state, r, done, _ = env.step(action)
            epi_return += r
            env.render()
            # time.sleep(0.05)
            state = next_state
        print("Total reward of visualized episode: {}".format(epi_return))

    def get_eval_epi_reward(self):
        with torch.no_grad():
            s = env.reset()
            done = False
            reward = 0
            while not done:
                s_tens = torch.tensor(s, dtype=torch.float).to(self.device)
                a_tens = self.actor(s_tens)
                a = a_tens.detach().numpy()
                s_prime, r, done, _ = env.step(a)
                self.ReplayBuffer.add_one(s, a, r, s_prime, not done)
                s = s_prime
                reward += r
            return reward

    def train(self, num_steps):
        """
        Train the policy for the given number of iterations
        :param num_steps:The number of steps to train the policy for
        """
        for t in range(num_steps):
            self.store_transitions(self.batch_size_generate, randomize_action=False)

            s_batch, a_batch, r_batch, s_prime_batch, not_done_batch = self.ReplayBuffer.sample(self.batch_size_sample)

            assert s_batch.shape[1] == self.state_dim
            assert a_batch.shape[1] == self.action_dim
            assert r_batch.shape[1] == 1
            assert s_prime_batch.shape[1] == self.state_dim
            assert not_done_batch.shape[1] == 1

            self.update_critic(s_batch, a_batch, r_batch, s_prime_batch, not_done_batch)

            if t % self.policy_update_freq == 0:
                self.update_actor(s_batch)
                self.update_target_networks()

            if t % 500 == 0:
                self.log(self.get_eval_epi_reward(), "eval_epi_return")

                self.print_update(t)

        self.make_graphs()


if __name__ == "__main__":
    # Define the environment
    # env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    # env = gym.make("Reacher-v2")
    env = gym.make("Walker2d-v2")
    # env = gym.make("InvertedPendulum-v2")
    # env = gym.make("Pendulum-v1")
    # env = gym.make("CartPole-v1")

    ddpg_object = DDPG(
        env,
        critic_lr=3e-4,
        actor_lr=3e-4,
        gamma=0.99,
        batch_size_sample=256,
        batch_size_generate=1,
        policy_update_freq=3,  # number of critic updates per actor update
        tau=5e-3,  # how far to step targets towards trained policies
        exploration_noise=0.1,  # noise to add to policy output in rollouts
        a_tilde_noise=0.2  # noise to add to actions for q function estimates
    )

    # profiler = Profiler()
    # profiler.start()

    # Train the policy
    ddpg_object.train(int(1e5))

    # profiler.stop()
    # profiler.print()

    ddpg_object.visualize_rollout()


import torch
import gym
from model import Model


def calculate_target(final_v, rewards, gamma):
    target = []
    for reward in reversed(rewards):
        final_v = reward + gamma * final_v
        target.append(final_v)
    target.reverse()
    return torch.as_tensor(target)


def train(env_name, shared_model, max_no_steps, max_no_episodes, shared_optimizer):
    env = gym.make(env_name)
    observation_no = env.observation.shape[0]
    action_no = env.action_space.n
    model = Model(observation_no, action_no)


    for i in range(max_no_episodes):
        state = env.reset()
        state = torch.as_tensor(state)
        ep_tot_rew = 0
        ep_rew, ep_act, ep_state = [], [], []
        while True:
            action = model.action(state)
            #n_state, rew, done _ = env.step(action)
            ep_tot_rew + = rew



if __name__ == '__main__':
    final_v = 6
    rewards = torch.Tensor([1,2,3,1,5])+torch.Tensor([1,2,3,3,3])
    print(rewards)
    print(calculate_target(final_v, rewards, 0.9))
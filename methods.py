import torch
import gym
from model import Model
import torch.multiprocessing as mp

def calculate_target(final_v, rewards, gamma):
    target = []
    for reward in reversed(rewards):
        final_v = reward + gamma * final_v
        target.append(final_v)
    target.reverse()
    return torch.as_tensor(target)


def train(env_name, shared_model, max_no_steps, max_no_episodes, shared_optimizer, process_id, global_ep_no):
    env = gym.make(env_name)
    observation_no = env.observation.shape[0]
    action_no = env.action_space.n
    model = Model(observation_no, action_no)

    for i in range(max_no_episodes):
        model.load_state_dict(shared_model.state_dict())
        state = env.reset()
        time_step = 0
        #state = torch.as_tensor(state)
        ep_tot_rew = 0
        ep_rew, ep_act, ep_state = [], [], []
        while True:
            time_step = time_step + 1
            action = model.action(torch.as_tensor(state))
            n_state, rew, done, _ = env.step(action)
            ep_tot_rew  = ep_tot_rew + rew
            ep_rew.append(rew)
            ep_act.append(action)
            ep_state.append(state)
            state = n_state
            if time_step > max_no_steps or done:
                if done:
                    targets = calculate_target(0, ep_rew, gamma=0.9)
                else:
                    _, critic_value = model.forward(torch.as_tensor(n_state))
                    targets = calculate_target(critic_value, ep_rew, gamma=0.9)
                loss = model.calculate_loss(targets=targets,states=torch.as_tensor(ep_state), actions=torch.as_tensor(ep_act))
                shared_optimizer.zero_grad()
                loss.backward()
                for g_net, l_net in zip(shared_model.parameters(), model.parameters()):
                    g_net._grad = l_net.grad
                shared_optimizer.step()
                with global_ep_no.get_lock():
                    global_ep_no.value += 1
                print("Global Episode No: {} Process ID: {} Episode No: {} Total Reward: {}".format(global_ep_no.value,process_id, i, ep_tot_rew))
                break



    # final_v = 6
    # rewards = torch.Tensor([1,2,3,1,5])+torch.Tensor([1,2,3,3,3])
    # print(rewards)
    # print(calculate_target(final_v, rewards, 0.9))
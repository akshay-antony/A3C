#import gym
import torch
import  torch.multiprocessing as mp
import argparse
from model import Model
from shared_optimizer import SharedAdam
from methods import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="selecting the gym env")
    parser.add_argument("-env_name",default="CartPole-v0")
    args = parser.parse_args()
    num_processes = mp.cpu_count()

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    shared_model = Model(env.observation_space.shape[0], env.action_space.n)
    shared_model.share_memory()
    global_ep_no = mp.Value('i', 0)
    shared_optimizer = SharedAdam(shared_model.parameters())
    processes = []

    max_no_steps = 100
    max_no_episodes = 50
    for i in range(num_processes):
        p = mp.Process(target= train, args=(env_name, shared_model, max_no_steps, max_no_episodes, shared_optimizer, i, global_ep_no))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()



    # li = []
    # for i in range(10):
    #     li.append(torch.rand(4))
    # li = torch.stack(li)
    # m = Model(4,3)
    # act,_ =m.forward(li)
    # #print(act)
    # act = Categorical(logits=act)
    # action = act.sample()
    # loss= -act.log_prob(action)
    # x=[]
    # for i in range(10):
    #     x.append(2)
    # x=torch.as_tensor(x)
    # print(loss*x)
    # print(loss*x.detach().squeeze())
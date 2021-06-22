#import gym
import torch
from torch import multiprocessing as mp
import argparse
from model import Model
from torch.distributions.categorical import Categorical

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="selecting the gym env")
    parser.add_argument("-env_name",default="CartPole-v0")
    args = parser.parse_args()
    print(mp.gpu_count())


    #
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
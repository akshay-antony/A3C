import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1_actor = nn.Linear(input_size, 128)
        self.l2_actor = nn.Linear(128, output_size)
        self.l1_critic = nn.Linear(input_size, 128)
        self.l2_critic = nn.Linear(128, 1)
        init_weights(self.l1_actor)
        init_weights(self.l2_actor)
        init_weights(self.l1_critic)
        init_weights(self.l2_critic)

    def forward(self, state):
        action_values = self.l2_actor(torch.tanh(self.l1_actor(state)))
        critic_values = self.l2_critic(torch.tanh(self.l1_critic(state)))
        return action_values, critic_values

    def action(self, state):
        action_values,_ = self.forward(state)
        action = Categorical(logits=action_values) #verify if sigmoid required
        return action.sample().numpy()             #check if numpy required

    def calculate_loss(self, targets, states, actions):
        action_values, critic_values = self.forward(states)
        #action = self.ac
        adv=  targets - critic_values
        critic_loss = adv*adv
        m = Categorical(logits=action_values)
        actor_loss = -m.log_prob(actions)*adv
        loss = actor_loss + critic_loss
        return loss.mean()

    def train_step(self, loss, optimizer):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def init_weights(layer):
    nn.init.normal_(layer.weight, mean=0, std=.1)
    nn.init.constant_(layer.bias, 0)
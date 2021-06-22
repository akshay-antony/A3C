import torch
from torch.optim import Adam


class SharedAdam(Adam):
    def __init__(self, params, lr=.0001, betas=(0.9,0.99), eps=1e-8, weight_decay=0 ):
        super().__init__(params=params,lr=lr, betas=betas, eps=eps, weight_decay= weight_decay)
        #initialize the state wrt to parameters of different models running in parallel and they all share the
        #exp_avg, exo_avg_sq which carries the momentum for optimization

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

import torch
import math


class LrburstScheduler:
    def __init__(self, eta_max, eta_min, T_max=150, H="cyclic"):
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max = T_max
        self.H = H
        self.T_cur = 0

    def get_lr(self):
        if self.H == "cyclic":
            if self.T_cur == self.T_max:
                self.T_cur = 1
            else:
                self.T_cur += 1

        phi = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(self.T_cur * math.pi / self.T_max))
        self.T_cur += 1
        return phi


class SGDWithMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGDWithMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = p.data - p.data * weight_decay
                d_p = p.grad.data
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                state = self.state[p]
                state['momentum_buffer'] = momentum * state['momentum_buffer'] +  d_p
                p.data = p.data - group['lr'] * state['momentum_buffer']

        return p.data
     
        

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-4, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid betas: {}, {}".format(betas[0], betas[1]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']

                m.mul_(beta1).add_(grad)
                v.mul_(beta2).add_(grad, grad**2)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                p.data.addcdiv_(-group['lr'], m_hat, v_hat.sqrt().add_(group['eps']))
                
                # Weight decay (L2 regularization)
                if group['weight_decay'] != 0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return p.data



class Adaburst(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-4, weight_decay=0, dropout=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 value: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 value: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.A = torch.optim.AdamW(params, lr=lr, betas=betas,eps=eps, weight_decay=weight_decay)
        self.S = torch.optim.SGD(params,lr=lr, eps=eps,weight_decay=weight_decay)
        self.Lrburst = LrburstScheduler(eta_max=1e-3, eta_min=1e-9):
        super(Adaburst, self).__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['step'] += 1
                if group['weight_decay'] != 0:
                    p.data = p.data - p.data * group['weight_decay']

                p.data = p.data - p.data * weight_decay

                state['m'] = group['betas'][0] * state['m'] + grad
                state['v'] = group['betas'][1] * state['v'] + grad ** 2
                
                m_hat = state['m'] / (1 - group['betas'][0] ** state['step'])
                v_hat = state['v'] / (1 - group['betas'][1] ** state['step'])
                
                A = self.A.step()
                lr2 = group['lr'] * group['eps'] * self.Lrburst.get_lr()
                S = self.S.step()
                
                p.data = p.data - lr1 * A - lr2 * S


        return p.data
        



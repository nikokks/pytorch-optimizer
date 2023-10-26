import torch

class FastAdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, delta=1e-4):
        defaults = dict(lr=lr, betas=betas, eps=eps, delta=delta)
        super(FastAdaBelief, self).__init__(params, defaults)

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

                state['step'] += 1
                t = state['step']
                alpha_t = group['lr'] / t

                if 'm' not in state:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)

                m, v, max_v, delta = state['m'], state['v'], state['s'], group['delta']

                m.mul_(group['betas'][0]).add_(1 - group['betas'][0], grad)
                v.mul_(group['betas'][1]).addcmul_(1 - group['betas'][1], grad - m, grad - m)

                max_v = torch.max(max_v, v)

                st_hat = max_v
                St_hat = torch.diag(st_hat) + delta / t * torch.eye(st_hat.shape[0])
                test = torch.pinverse(St_hat) @ m
                
                inside_term = p.data - alpha_t * test
                p.data = inside_term

        return loss

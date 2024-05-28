import torch
import torch.optim as optim


class MyAdam(optim.Optimizer):
    def __init__(
            self, params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            use_bias_correction=True,
            epsilon_inside_sqrt=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.use_bias_correction = use_bias_correction
        self.epsilon_inside_sqrt = epsilon_inside_sqrt
        super(MyAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with closure() as loss:
                pass

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('CustomAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                # Compute bias-corrected second raw moment estimate
                bias_correction2 = 1 - beta2 ** state['step']
                if not self.use_bias_correction:
                    bias_correction1 = 1.
                    bias_correction2 = 1.

                if self.epsilon_inside_sqrt:
                    denom = (exp_avg_sq / bias_correction2).add_(group['eps']).sqrt()  # inside!
                else:
                    denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(group['eps'])  # outside!

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
import torch
from torch import nn
from torch.autograd import grad


class Generator(nn.Module):
    def __init__(self, num_latent):
        super(Generator, self).__init__()
        self.num_latent = num_latent

    def sample(self, num_samples=1):
        z = torch.zeros(num_samples, self.num_latent).normal_().to(self.device)
        x = self.forward(z)
        return x

    def to(self, device=None):
        self.device = device
        return super().to(device=device)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        raise NotImplementedError()

    def get_penalty(self, x_true, x_gen):
        x_true = x_true.view_as(x_gen)
        alpha = torch.rand((len(x_true),)+(1,)*(x_true.dim()-1))
        if x_true.is_cuda:
            alpha = alpha.cuda(x_true.get_device())
        x_penalty = alpha*x_true + (1-alpha)*x_gen
        x_penalty.requires_grad_()
        p_penalty = self.forward(x_penalty)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True, retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1)**2).mean()

        return penalty


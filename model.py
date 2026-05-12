import torch
import torch.nn as nn

# ---------- Custom Linear Layer ----------
class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        return x @ w.T + b

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = grad_output @ w
        grad_w = grad_output.T @ x
        grad_b = grad_output.sum(0)
        return grad_x, grad_w, grad_b

# ---------- SiLU Activation ----------
class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sigmoid = torch.sigmoid(x)
        ctx.save_for_backward(x)
        return sigmoid * x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid + x * sigmoid * (1 - sigmoid))

# ---------- MSE Loss ----------
class MSELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target):
        diff = pred - target
        ctx.save_for_backward(diff)
        return (diff ** 2).mean()

    @staticmethod
    def backward(ctx, grad_output):
        diff, = ctx.saved_tensors
        n = diff.numel()
        # Gradient for pred (the first argument) is used; target gradient is not used during training
        return grad_output * 2 * diff / n, None

# ---------- Custom SGD Optimizer ----------
class CustomSGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

# ---------- Neural Network ----------
class SineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 1) * 0.1)
        self.b1 = nn.Parameter(torch.randn(16) * 0.1)
        self.w2 = nn.Parameter(torch.randn(16, 16) * 0.1)
        self.b2 = nn.Parameter(torch.randn(16) * 0.1)
        self.w3 = nn.Parameter(torch.randn(1, 16) * 0.1)
        self.b3 = nn.Parameter(torch.randn(1) * 0.1)

    def forward(self, x):
        h = SiLU.apply(CustomLinear.apply(x, self.w1, self.b1))
        h = SiLU.apply(CustomLinear.apply(h, self.w2, self.b2))
        return CustomLinear.apply(h, self.w3, self.b3)
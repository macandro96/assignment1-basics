import torch


def cross_entropy(predicted: torch.Tensor, targets: torch.Tensor):
    """Computes cross-entropy loss.
    l_i = -log softmax(pred_i)[x_{i+1}]

    Args:
        - predicted (torch.Tensor): logits of size bsz x |V|
        - targets (torch.Tensor): target indicies of size bsz
    """
    log_softmaxs = log_softmax(predicted)  # bsz x |V|
    ce = -log_softmaxs.gather(-1, targets.unsqueeze(-1))
    return torch.mean(ce)


def log_softmax(logits: torch.Tensor):
    """Log softmax

    Log softmax:
        -> log (e^(x-max) / sum(e^(x_{i} - max)))
        -> log (e^(x-max)) - log(sum((e^x_{i} - max)))
        -> x - max - log[sum(e^x_{i} - max)]
    Args:
        - logits (torch.Tensor): logits of size bsz x |V|
    """
    # max
    max_elements, _ = torch.max(logits, dim=-1, keepdim=True)  # bsz x 1
    exps = torch.exp(logits - max_elements)
    sum_exps = torch.sum(exps, dim=-1, keepdim=True)  # bsz x 1
    log_exps = torch.log(sum_exps)  # bsz x |V|
    return logits - max_elements - log_exps  # bsz x |V|


def gradient_clipping(model_params, max_l2_norm: float, eps: float = 1e-6):
    grads = [param.grad for param in model_params if param.grad is not None]
    grad_norm = (sum([(grad**2).sum() for grad in grads])) ** (0.5)

    if grad_norm > max_l2_norm:
        coeff = max_l2_norm / (grad_norm + eps)
    else:
        return
    for param in model_params:
        if param.grad is not None:
            param.grad = param.grad * coeff


def softmax(x: torch.Tensor, dim: int):
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    numerator = torch.exp(x - max_vals)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np

def my_softmax(input, axis=-1):
    return F.softmax(input, dim=axis)

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Borrowed from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Borrowed from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)   # g
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)             # y = logits/pi + g
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Borrowed from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def kl_categorical(input, target, num_nodes, eps=1e-16):
    """
    Parameters
    ----------
    input : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
        Predicted edge type probabilities.
    target : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
        Target edge type probabilities (Prior distribution)
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        The KL divergence value.
    """

    kl_div = input * (torch.log(input + eps) - torch.log(target + eps))
    return kl_div.sum() / (num_nodes * input.size(0)) # shape (scalar)
    
def kl_categorical_uniform(input, num_nodes, add_const=False, eps=1e-16):
    """
    Parameters
    ----------
    input : torch.Tensor, shape (batch_size, n_edges, n_edge_types)
        Predicted edge type probabilities.
    num_nodes : int
        Number of nodes in the graph.
    add_const : bool, optional
        If True, adds a constant term to the KL divergence.
    eps : float, optional
        Small value to avoid log(0).

    Returns
    -------
    torch.Tensor (Scalar)
        The KL divergence value for uniform target (prior) distribution.
    """
    num_distributions = input.shape[0] * input.shape[1]
    kl_div = input * torch.log(input + eps)
    kl_sum = kl_div.sum()
    if add_const:
        kl_sum += torch.log(input.size(-1)) * num_distributions

    return kl_sum / (num_nodes * input.size(0)) # shape (scalar)

def nll_gaussian(pred, target, variance):
    """

    Parameters
    ----------
    pred : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
    target : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
    variance: torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)

    Returns
    -------
    torch.Tensor (Scalar)
        The negative log likelihood (NLL) of the Gaussian distribution.
        The NLL is averaged over the total number of nodes in the batch
    """

    term_1 = ((pred - target) ** 2 / (2 * variance))
    term_2 = 0.5 * torch.log(2 * torch.pi * variance)
    nll = term_1 + term_2       # shape (batch_size, n_nodes, n_timesteps-1, n_dim)

    return nll.sum() / (target.size(0) * target.size(1)) # shape (scalar)
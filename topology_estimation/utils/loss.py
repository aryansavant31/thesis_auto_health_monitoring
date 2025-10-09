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
    # move target to the device of input
    target = target.to(input.device)

    # kl_per_edge = input * (torch.log(input + eps) - torch.log(target + eps))
    kl_per_edge = torch.sum(input * (torch.log(input + eps) - torch.log(target + eps)), dim=-1)  # shape (batch_size, n_edges)
    mean_kl_per_edge = kl_per_edge.mean()  # shape (scalar)
    # mean computed over n_edges * batch_size

    kl_per_sample = kl_per_edge.sum(dim=1)  # shape (batch_size)
    mean_kl_per_sample = kl_per_sample.mean()  # shape (scalar)
    # mean computed over all samples in the batch

    return mean_kl_per_edge, mean_kl_per_sample

    
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
    # num_distributions = input.shape[0] * input.shape[1]
    # kl_div = input * torch.log(input + eps)
    # kl_sum = kl_div.sum() # sums over n_batches x n_edges
    # if add_const:
    #     kl_sum += np.log(input.size(-1)) * num_distributions

    # return kl_sum / (num_nodes * input.size(0)) # shape (scalar)
    n_edge_types = input.size(-1)

    kl_per_edge = torch.sum(input * torch.log(input + eps), dim=-1)  
    if add_const:
        kl_per_edge += np.log(n_edge_types) # shape (batch_size, n_edges)

    # mean quantiles
    mean_kl_per_edge = kl_per_edge.mean()  # shape (scalar)
    # mean computed over n_edges * batch_size

    # per-sample
    kl_per_sample = kl_per_edge.sum(dim=1)  # shape (batch_size)
    mean_kl_per_sample = kl_per_sample.mean()  # shape (scalar)
    # mean computed over all samples in the batch

    return mean_kl_per_edge, mean_kl_per_sample


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

def smape(pred, target, is_per_node=False, eps=1):
    """
    smape: Symmetric Mean Absolute Percentage Error
    Parameters
    ----------
    pred : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
    target : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)

    Returns
    -------
    torch.Tensor (Scalar) or torch.Tensor (n_nodes,)
        The Mean Absolute Percentage Error (MAPE).
        If is_per_node is False, the MAPE is averaged over the total number of nodes in the batch.
        If is_per_node is True, the MAPE is computed per node and averaged over the batch.
    """
    eps = 0.1 * torch.mean(torch.abs(target))  # 10% of average magnitude
    sape = torch.abs(pred - target) / (((torch.abs(pred) + torch.abs(target)) / 2) + eps)  # shape (batch_size, n_nodes, n_timesteps-1, n_dim)
    

    if is_per_node:
        smape_per_node = sape.mean(dim=(0, 2, 3))  # shape (n_nodes,)
        return smape_per_node
    else:
        smape = sape.mean()  # shape (scalar)
        return smape
    
def correlation_loss(pred, target, is_per_node=False, eps=1e-8):
    
    if is_per_node:
        n_nodes = pred.size(1)
        corr_per_node = torch.zeros(n_nodes).to(pred.device)
        for i in range(n_nodes):
            vx = pred[:, i, :, :] - pred[:, i, :, :].mean()
            vy = target[:, i, :, :] - target[:, i, :, :].mean()
            corr_per_node[i] = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)) + eps)
        return 1 - corr_per_node
    else:
        vx = pred - pred.mean()
        vy = target - target.mean()
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2) * torch.sum(vy**2)) + eps)
        return 1 - corr
    
# def smspe(pred, target, is_per_node=False, eps=1):
#     """
#     smspe: Symmetric Mean Squared Percentage Error

#     Parameters
#     ----------
#     pred : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
#     target : torch.Tensor, shape (batch_size, n_nodes, n_timesteps-1, n_dim)
#     is_per_node : bool, optional
#         If True, computes the SMSPE per node and averages over the batch.
#         If False, computes the SMSPE averaged over the total number of nodes in the batch.
#     eps : float, optional
#         Small value to avoid division by zero.
#         Default is 1.

#     Returns
#     -------
#     torch.Tensor (Scalar) or torch.Tensor (n_nodes,)
#         The Mean Squared Percentage Error (MSPE).
#         If is_per_node is False, the MSPE is averaged over the total number of nodes in the batch.
#         If is_per_node is True, the MSPE is computed per node and averaged over the batch.
#     """
#     eps = 0.1 * torch.mean(torch.abs(target))  # 10% of average magnitude
#     sspe = ((pred - target) ** 2) / (((torch.abs(pred) + torch.abs(target)) / 2 + eps) ** 2)  # shape (batch_size, n_nodes, n_timesteps-1, n_dim)
    
#     if is_per_node:
#         mspe_per_node = sspe.mean(dim=(0, 2, 3))  # shape (n_nodes,)
#         return mspe_per_node
#     else:
#         mspe = sspe.mean()  # shape (scalar)
#         return mspe
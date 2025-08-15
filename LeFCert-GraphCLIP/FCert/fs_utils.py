import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
import time
from torch_geometric.data import Data 

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x, y):
    '''
    Compute cosine distance between two tensors
    '''
    similarity = F.cosine_similarity(x.unsqueeze(1), y, dim=-1)
    distance = 1 - similarity
    return distance


def cosine_similarity(x, y):
    '''
    Compute cosine similarity between two tensors
    '''
    similarity = F.cosine_similarity(x.unsqueeze(1), y, dim=-1)
    return similarity

def l2_dist(x, y):
    '''
    Compute L2 distance between two tensors
    '''
    return torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

def l1_dist(x, y):
    '''
    Compute L1 distance between two tensors
    '''
    return torch.cdist(x, y, p=1.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

def get_distances(opt, input, target, n_support):
    '''
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # Get the indices of the first n_support samples for class c
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # Get unique classes
    classes = torch.unique(target_cpu)

    n_classes = len(classes)
    # Calculate the number of query samples per class
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # Get the indices of support samples for each class
    support_idxs = list(map(supp_idxs, classes))

    # Stack support samples and flatten
    support_samples = torch.cat([input_cpu[idx_list] for idx_list in support_idxs])

    # Get the indices of query samples for each class
    query_idxs = torch.cat([target_cpu.eq(c).nonzero()[n_support:] for c in classes]).squeeze()

    # Get query samples
    query_samples = input_cpu[query_idxs]

    # Compute distances based on the specified metric
    if opt.metric_type == "euclidean":
        dists = euclidean_dist(query_samples, support_samples)
    elif opt.metric_type == "cosine":
        dists = cosine_dist(query_samples, support_samples)
    elif opt.metric_type == "l1":
        dists = l1_dist(query_samples, support_samples)
    elif opt.metric_type == "l2":
        dists = l2_dist(query_samples, support_samples)
    else:
        raise ValueError(f"Unsupported metric type: {opt.metric_type}")

    # Create target indices for query samples
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    return dists, target_inds

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

from torch_geometric.data import Data

'''Get cora node embedding from graphclip model'''
_global_embeddings = None

def get_feature(opt, model, batch, node_ids):
    global _global_embeddings
    if opt.use_subgraph:
        with torch.no_grad():
            subgraph = batch.to(opt.device)
            embeddings = model.encode_graph(subgraph)
            center_idx = subgraph.root_n_index 
            return embeddings  #[C*(K+num_query),feature_dim]
    else:# Calculate the features of the entire image when it is called for the first time.
        if _global_embeddings is None:
            with torch.no_grad():
                full_graph = opt.graph_data.to(opt.device)
                _global_embeddings = model.encode_graph(full_graph)
        #return node embedding for batch data
        return _global_embeddings[node_ids.to(opt.device)]

def get_f_t_dict(opt, input,text_features, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.flatten(torch.stack([input_cpu[idx_list] for idx_list in support_idxs]), start_dim=0, end_dim=1)
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    if opt.metric_type =="cosine":
        dicts = cosine_dist(text_features.to('cpu'), support_samples)

    query_samples = input.to('cpu')[query_idxs]
    test_dicts=cosine_dist(text_features.to('cpu'), query_samples)
    return dicts,test_dicts

def get_f_f_dict(opt, input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.flatten(torch.stack([input_cpu[idx_list] for idx_list in support_idxs]), start_dim=0, end_dim=1)
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    #input_cpu[idx_list]
    if opt.metric_type =="cosine":
        dicts = cosine_dist(query_samples, support_samples)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    return dicts, target_inds


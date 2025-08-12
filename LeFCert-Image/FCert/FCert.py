import torch
import numpy as np
from .fs_utils import *
import clip
import itertools
from tqdm import tqdm

def FCert(opt, test_dataloader, model,clip_k=4,T=10,certify_type = "group"):
    device = opt.device
    model = model.to(device)
    num_batches=test_dataloader.__len__()
    avg_acc = np.zeros((num_batches, T))
    for batch_id, batch in enumerate(test_dataloader):
        if batch_id == num_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_output = get_feature(opt, model, x)
        dist, target_inds = get_distances(opt, model_output, target=y,
                                          n_support=opt.K)
        dist = torch.reshape(dist, (dist.size(0), opt.C, opt.K))
        sorted_dist, _ = torch.sort(dist, dim=-1)
        # certify for group attack
        if certify_type == "group":
            clipped_dist = sorted_dist[:, :, clip_k:sorted_dist.shape[2] - clip_k]
            clipped_mean = torch.mean(clipped_dist, dim=-1)
            all_clipped_mean_upper = []
            all_clipped_mean_lower = []

            for t in range(min(clip_k+1, T+1)):
                clipped_dist_upper = sorted_dist[:, :, clip_k + t:sorted_dist.shape[2] - clip_k + t]
                clipped_dist_lower = sorted_dist[:, :, clip_k - t:sorted_dist.shape[2] - clip_k - t]
                clipped_mean_upper = torch.mean(clipped_dist_upper, dim=-1)
                clipped_mean_lower = torch.mean(clipped_dist_lower, dim=-1)
                all_clipped_mean_upper.append(clipped_mean_upper)
                all_clipped_mean_lower.append(clipped_mean_lower)

        for t in range(T):
            if t>clip_k:#certification fails if t>k'
                avg_acc[batch_id,t] = 0
            else:
                all_preds = []
                #Suppose the attacker modifies t1 data samples in the ground-truth class, and t2 data samples in the non-ground-truth class
                #Consider all combinations of t1 and t2
                for t1 in range(t+1):
                    t2 = t -t1
                    bounds = torch.zeros_like(clipped_mean)
                    for i in range(clipped_mean.shape[0]):
                        for j in range(clipped_mean.shape[1]):
                            if torch.flatten(target_inds)[i] == j:
                                bounds[i][j] = all_clipped_mean_upper[t1][i][j]
                            else:
                                bounds[i][j] = all_clipped_mean_lower[t2][i][j]
                    preds = torch.argmin(bounds, dim = -1)
                    all_preds.append(preds)

                all_preds = torch.stack(all_preds, dim = 0)
                preds_poisoned = torch.zeros(all_preds.shape[1])
                all_preds = torch.transpose(all_preds, 0, 1)

                for i in range(all_preds.shape[0]):
                    pred = all_preds[i]
                    if torch.all(torch.eq(pred, pred[0])):
                        preds_poisoned[i] = pred[0]
                    else:
                        preds_poisoned[i] = -1
                correct_count = 0

                for i in range(torch.flatten(target_inds).shape[0]):
                    if preds_poisoned[i] == torch.flatten(target_inds)[i]:
                        correct_count+=1

                avg_acc[batch_id,t] = correct_count/torch.flatten(target_inds).shape[0]

    avg_acc = np.mean(avg_acc,axis=0)
    return avg_acc.tolist()

def box_bound(t, K, M, ff_sorted_dist,ft_sorted_dist,test_dicts,Lambda):
    #feature-feature in support set:
    ff_clipped_dist_upper = ff_sorted_dist[:, :, M + t:K]
    ff_clipped_dist_lower = ff_sorted_dist[:, :, 0:K - M - t]
    ff_clipped_sum_upper = torch.sum(ff_clipped_dist_upper, dim=-1)+2*(t-M)#max=2
    ff_clipped_sum_lower = torch.sum(ff_clipped_dist_lower, dim=-1)+0
    #feature-text in support set:
    ft_clipped_dist_upper = ft_sorted_dist[:, :, M + t:K]
    ft_clipped_dist_lower = ft_sorted_dist[:, :, 0:K - M - t]
    ft_clipped_mean_upper = (torch.sum(ft_clipped_dist_upper, dim=-1)+ 2*(t-M))/ (K - 2*M)#max=2
    ft_clipped_mean_lower = torch.sum(ft_clipped_dist_lower, dim=-1)/ (K - 2*M)

    clipped_mean_upper = ff_clipped_sum_upper + (Lambda * ft_clipped_mean_upper)* test_dicts
    clipped_mean_lower = ff_clipped_sum_lower + (Lambda * ft_clipped_mean_lower)* test_dicts
    return clipped_mean_upper,clipped_mean_lower

def normal_bound(t, K, M, ff_sorted_dist, ft_sorted_dist, test_dicts,Lambda):
    ff_clipped_dist_upper = ff_sorted_dist[:, :, M + t:ff_sorted_dist.shape[2] - M + t]
    ff_clipped_dist_lower = ff_sorted_dist[:, :, M - t:ff_sorted_dist.shape[2] - M - t]
    ff_clipped_sum_upper = torch.sum(ff_clipped_dist_upper, dim=-1)
    ff_clipped_sum_lower = torch.sum(ff_clipped_dist_lower, dim=-1)
    # feature-text in support set:
    ft_clipped_dist_upper = ft_sorted_dist[:, :, M + t:ff_sorted_dist.shape[2] - M + t]
    ft_clipped_dist_lower = ft_sorted_dist[:, :, M - t:ff_sorted_dist.shape[2] - M - t]
    ft_clipped_mean_upper = torch.mean(ft_clipped_dist_upper, dim=-1)
    ft_clipped_mean_lower = torch.mean(ft_clipped_dist_lower, dim=-1)

    clipped_mean_upper = ff_clipped_sum_upper + (Lambda  * ft_clipped_mean_upper) * test_dicts
    clipped_mean_lower = ff_clipped_sum_lower + (Lambda  * ft_clipped_mean_lower) * test_dicts
    return clipped_mean_upper, clipped_mean_lower


def LeFCert(opt, test_dataloader, model,class_map, clip_k=4, T=10, certify_type="ind"):
    device = opt.device
    model = model.to(device)
    num_batches = test_dataloader.__len__()
    avg_acc = np.zeros((num_batches, T))
    for batch_id, batch in enumerate(test_dataloader):
        if batch_id == num_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        # get the label text
        classes = torch.unique(batch[1])
        label_text = [f"It is a photo of {class_map[class_id.item()]}" for class_id in classes]
        label_text_tokens = clip.tokenize(label_text).to(device)
        text_features = model.encode_text(label_text_tokens)  # t_k

        model_output = get_feature(opt, model, x)
        ff_dicts, target_inds = get_f_f_dict(opt, model_output, target=y, n_support=opt.K)
        ff_dicts = torch.reshape(ff_dicts, (ff_dicts.size(0), opt.C, opt.K))
        ft_dicts, test_dicts = get_f_t_dict(opt, model_output, text_features, target=y, n_support=opt.K)
        ft_dicts = torch.reshape(ft_dicts, (ft_dicts.size(0), opt.C, opt.K))

        ff_sorted_dist, _ = torch.sort(ff_dicts, dim=-1)
        ft_sorted_dist, _ = torch.sort(ft_dicts, dim=-1)

        batch_sizes = ft_dicts.size(0)
        C = opt.C
        K = opt.K

        # certify for group attack
        if certify_type == "group":
            all_clipped_mean_upper = []
            all_clipped_mean_lower = []
            for t in range(min(clip_k+1,T + 1)):
                clipped_mean_upper, clipped_mean_lower = normal_bound(t, K, clip_k, ff_sorted_dist,
                                                                      ft_sorted_dist, test_dicts,
                                                                      opt.Lambda)
                # else:
                #     clipped_mean_upper, clipped_mean_lower = box_bound(t, K, clip_k, ff_sorted_dist,
                #                                                        ft_sorted_dist, test_dicts, opt.Lambda)
                all_clipped_mean_upper.append(clipped_mean_upper)
                all_clipped_mean_lower.append(clipped_mean_lower)

        for t in range(T):
            if t>clip_k:#certification fails if t>k'
                avg_acc[batch_id,t] = 0
            else:
                all_preds = []
                #Suppose the attacker modifies t1 data samples in the ground-truth class, and t2 data samples in the non-ground-truth class
                #Consider all combinations of t1 and t2
                for t1 in range(t+1):
                    t2 = t -t1
                    bounds = torch.zeros(batch_sizes,C)
                    for i in range(batch_sizes):
                        for j in range(C):
                            if torch.flatten(target_inds)[i] == j:
                                bounds[i][j] = all_clipped_mean_upper[t1][i][j]
                            else:
                                bounds[i][j] = all_clipped_mean_lower[t2][i][j]
                    preds = torch.argmin(bounds, dim = -1)
                    all_preds.append(preds)

                all_preds = torch.stack(all_preds, dim = 0)
                preds_poisoned = torch.zeros(all_preds.shape[1])
                all_preds = torch.transpose(all_preds, 0, 1)

                for i in range(all_preds.shape[0]):
                    pred = all_preds[i]
                    if torch.all(torch.eq(pred, pred[0])):
                        preds_poisoned[i] = pred[0]
                    else:
                        preds_poisoned[i] = -1
                correct_count = 0

                for i in range(torch.flatten(target_inds).shape[0]):
                    if preds_poisoned[i] == torch.flatten(target_inds)[i]:
                        correct_count+=1

                avg_acc[batch_id,t] = correct_count/torch.flatten(target_inds).shape[0]
    avg_acc = np.mean(avg_acc, axis=0)
    return avg_acc


def normal_bound_merge(t, K, M, sorted_dist,num_query):
    assert sorted_dist.shape[2] == K, "The third dimension of sorted_dist must be equal to K"
    sorted_clipped_dist_upper = sorted_dist[:, :, M + t:sorted_dist.shape[2] - M + t]
    sorted_clipped_dist_lower = sorted_dist[:, :, M - t:sorted_dist.shape[2] - M - t]
    # Calculate the sum of the clipped distances
    clipped_mean_upper = torch.sum(sorted_clipped_dist_upper, dim=-1)
    clipped_mean_lower = torch.sum(sorted_clipped_dist_lower, dim=-1)
    return clipped_mean_upper, clipped_mean_lower


def LeFCertNew(opt, test_dataloader, model, class_map, clip_k=4, T=10, certify_type="ind"):
    device = opt.device
    model = model.to(device)
    num_batches = test_dataloader.__len__()
    avg_acc = np.zeros((num_batches, T))
    for batch_id, batch in enumerate(test_dataloader):
        if batch_id == num_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        # get the label text
        classes = torch.unique(batch[1])
        label_text = [f"It is a photo of {class_map[class_id.item()]}" for class_id in classes]
        label_text_tokens = clip.tokenize(label_text).to(device)
        text_features = model.encode_text(label_text_tokens)  # t_k

        model_output = get_feature(opt, model, x)
        ff_dicts, target_inds = get_f_f_dict(opt, model_output, target=y, n_support=opt.K)
        ff_dicts = torch.reshape(ff_dicts, (ff_dicts.size(0), opt.C, opt.K))
        ft_dicts, test_dicts = get_f_t_dict(opt, model_output, text_features, target=y, n_support=opt.K)
        ft_dicts = torch.reshape(ft_dicts, (ft_dicts.size(0), opt.C, opt.K))

        batch_sizes = ft_dicts.size(0) * opt.num_query  # C x query
        C = opt.C
        K = opt.K

        test_dicts_expanded = test_dicts.unsqueeze(-1)
        merged_dist = ff_dicts + (opt.Lambda / K) * ft_dicts * test_dicts_expanded
        merged_sorted_dist, _ = torch.sort(merged_dist, dim=-1)


        # certify for group attack
        if certify_type == "group":
            all_clipped_mean_upper = []
            all_clipped_mean_lower = []
            for t in range(min(clip_k + 1, T + 1)):
                clipped_mean_upper, clipped_mean_lower = normal_bound_merge(t, K, clip_k, merged_sorted_dist,opt.num_query)
                # else:
                #     clipped_mean_upper, clipped_mean_lower = box_bound(t, K, clip_k, ff_sorted_dist,
                #                                                        ft_sorted_dist, test_dicts, opt.Lambda)
                all_clipped_mean_upper.append(clipped_mean_upper)
                all_clipped_mean_lower.append(clipped_mean_lower)

        for t in range(T):
            if t > clip_k:  # certification fails if t>k'
                avg_acc[batch_id, t] = 0
            else:
                all_preds = []
                # Suppose the attacker modifies t1 data samples in the ground-truth class, and t2 data samples in the non-ground-truth class
                # Consider all combinations of t1 and t2
                for t1 in range(t + 1):
                    t2 = t - t1
                    bounds = torch.zeros(batch_sizes, C)
                    for i in range(batch_sizes):
                        for j in range(C):
                            if torch.flatten(target_inds)[i] == j:
                                bounds[i][j] = all_clipped_mean_upper[t1][i][j]
                            else:
                                bounds[i][j] = all_clipped_mean_lower[t2][i][j]
                    preds = torch.argmin(bounds, dim=-1)
                    all_preds.append(preds)

                all_preds = torch.stack(all_preds, dim=0)
                preds_poisoned = torch.zeros(all_preds.shape[1])
                all_preds = torch.transpose(all_preds, 0, 1)

                for i in range(all_preds.shape[0]):
                    pred = all_preds[i]
                    if torch.all(torch.eq(pred, pred[0])):
                        preds_poisoned[i] = pred[0]
                    else:
                        preds_poisoned[i] = -1
                correct_count = 0

                for i in range(torch.flatten(target_inds).shape[0]):
                    if preds_poisoned[i] == torch.flatten(target_inds)[i]:
                        correct_count += 1

                avg_acc[batch_id, t] = correct_count / torch.flatten(target_inds).shape[0]
    avg_acc = np.mean(avg_acc, axis=0)
    return avg_acc



def traverse_optimal_median(sorted_dist, M, t, Lr, mode="max",aggr="sum"):
    """
    Find the maximum or minimum median of a sorted distance list by traversing all combinations of t modifications.
    Args:
    - sorted_dist (torch.Tensor): Sorted list of distances (1D tensor).
    - t (int): Number of elements the attacker can modify.
    - Lr (float): Maximum range of modification for each element.
    - mode (str): "max" to maximize the median, "min" to minimize the median.
    - aggr (str): Aggregation method for median calculation, either "sum" or "mean".
    Returns:
    - float: Maximum or minimum median after modifications.
    """
    C, B, K = sorted_dist.shape # C is the class number, B is the batch size, K is the shot number
    optimal_median = torch.full((C, B), float('-inf') if mode == "max" else float('inf'), device=sorted_dist.device)

    if t == 0:
        modified_dist = sorted_dist.clone()
        # Sort the modified tensor along the last dimension
        modified_dist, _ = torch.sort(modified_dist, dim=-1)
        # Remove the largest M and smallest M elements
        trimmed_dist = modified_dist[:, :, M:K - M]
        # Calculate the median for each [C, B] slice
        if aggr == "sum":
            median = torch.sum(trimmed_dist, dim=-1)
        elif aggr == "mean":
            median = torch.mean(trimmed_dist, dim=-1)

        return median

    # Generate all combinations of t indices for the last dimension (K)
    for indices in itertools.combinations(range(K), t):#C_k_t
        # Clone the original tensor to avoid modifying it directly
        modified_dist = sorted_dist.clone()

        # Apply the modification to the selected indices
        for idx in indices:
            if mode == "max":
                modified_dist[:, :, idx] += Lr
            elif mode == "min":
                modified_dist[:, :, idx] -= Lr

        # Sort the modified tensor along the last dimension
        modified_dist, _ = torch.sort(modified_dist, dim=-1)

        # Remove the largest M and smallest M elements
        trimmed_dist = modified_dist[:, :, M:K - M]

        # Calculate the median for each [C, B] slice
        if aggr == "sum":
            median = torch.sum(trimmed_dist, dim=-1)
        elif aggr == "mean":
            median = torch.mean(trimmed_dist, dim=-1)

        # Update the optimal median
        if mode == "max":
            optimal_median = torch.max(optimal_median, median)
        elif mode == "min":
            optimal_median = torch.min(optimal_median, median)

    return optimal_median


def get_optimal_Median(t,sorted_dist, Lr,mode):
    # it is not sure whether it is optimal
    n = sorted_dist.size(0)  # Number of elements in the list
    modified_dist = sorted_dist.clone()  # Clone the original tensor to avoid modifying it directly
    modified_indices = set()  # Track indices that have been modified

    for _ in range(t):
        if n % 2 == 1:  # Odd-length list
            median_idx = n // 2
        else:  # Even-length list
            median_idx1 = n // 2 - 1
            median_idx2 = n // 2

            # Test modifying both median elements
            if median_idx1 not in modified_indices and modified_dist[median_idx1] + Lr > modified_dist[median_idx2]:
                median_idx = median_idx1
            elif median_idx2 not in modified_indices:
                median_idx = median_idx2
            else:
                break  # Stop if neither median can be modified

        # Check if the median element can be modified
        if median_idx not in modified_indices:
            modified_dist[median_idx] += Lr
            modified_indices.add(median_idx)
        else:
            # Choose the element left to the median if the median is already modified
            left_idx = median_idx - 1
            if left_idx >= 0 and left_idx not in modified_indices:
                if modified_dist[left_idx] + Lr <= modified_dist[median_idx]:
                    modified_dist[left_idx] += Lr
                    modified_indices.add(left_idx)
                else:
                    break  # Stop if the left element cannot further enlarge the median
            else:
                break  # Stop if no valid element can be modified

        # Resort the list after modification
        modified_dist = torch.sort(modified_dist)[0]

        # Return the final median
    if n % 2 == 1:  # Odd-length list
        median_idx = n // 2
    else:  # Even-length list
        median_idx = n // 2 - 1 if modified_dist[n // 2 - 1] > modified_dist[n // 2] else n // 2

    return modified_dist[median_idx].item()

def get_bound_median(t,M,sorted_dist, Lr,aggr="sum"):
    if t <= M:
        clipped_dist_upper = sorted_dist[:, :, M + t:sorted_dist.shape[2] - M + t]
        clipped_dist_lower = sorted_dist[:, :, M - t:sorted_dist.shape[2] - M - t]
        clipped_dist_upper = torch.min(sorted_dist[:, :, M:sorted_dist.shape[2] - M] + Lr,
                                          clipped_dist_upper)
        clipped_dist_lower = torch.max(sorted_dist[:, :, M:sorted_dist.shape[2] - M] - Lr,
                                          clipped_dist_lower)

    else:
        clipped_dist_upper = sorted_dist[:, :, M:sorted_dist.shape[2] - M] + Lr
        clipped_dist_lower = sorted_dist[:, :, M:sorted_dist.shape[2] - M] - Lr
    if aggr == "sum":
        clipped_dist_upper = torch.sum(clipped_dist_upper, dim=-1)
        clipped_dist_lower = torch.sum(clipped_dist_lower, dim=-1)
    elif aggr == "mean":
        clipped_dist_upper = torch.mean(clipped_dist_upper, dim=-1)
        clipped_dist_lower = torch.mean(clipped_dist_lower, dim=-1)
    return clipped_dist_upper, clipped_dist_lower

def Liptchz_bound(t, K, M, ff_sorted_dist, ft_sorted_dist, test_dicts,Lr, Lambda):
    #----upper bound the median by all increase/descrease Lr to the median element
    # ff_clipped_sum_upper,ff_clipped_sum_lower=get_bound_median(t, M, ff_sorted_dist, Lr, aggr="sum")
    # ft_clipped_mean_upper,ft_clipped_mean_lower=get_bound_median(t, M, ft_sorted_dist, Lr, aggr="mean")
    #----upper bound the median by traversing all combinations of t modifications
    ff_clipped_sum_upper = traverse_optimal_median(ff_sorted_dist,M, t, Lr, mode="max",aggr="sum")
    ff_clipped_sum_lower = traverse_optimal_median(ff_sorted_dist, M, t, Lr, mode="min", aggr="sum")
    ft_clipped_mean_lower = traverse_optimal_median(ft_sorted_dist,M, t, Lr, mode="min",aggr="mean")
    ft_clipped_mean_upper = traverse_optimal_median(ft_sorted_dist,M, t, Lr, mode="max",aggr="mean")

    clipped_mean_upper = ff_clipped_sum_upper + (Lambda * ft_clipped_mean_upper) * test_dicts
    clipped_mean_lower = ff_clipped_sum_lower + (Lambda * ft_clipped_mean_lower) * test_dicts
    return clipped_mean_upper, clipped_mean_lower



def LeFCertL(opt, test_dataloader, model, class_map, clip_k=4, T=10, certify_type="group"):
    device = opt.device
    model = model.to(device)
    num_batches = test_dataloader.__len__()
    avg_acc = np.zeros((num_batches, T))

    for batch_id, batch in enumerate(tqdm(test_dataloader)):
        if batch_id == num_batches:
            break

        # Get the label text
        classes = torch.unique(batch[1])
        label_text = [f"It is a photo of {class_map[class_id.item()]}" for class_id in classes]
        label_text_tokens = clip.tokenize(label_text).to(device)
        text_features = model.encode_text(label_text_tokens)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        x, y = batch
        x, y = x.to(device), y.to(device)

        model_output = get_feature(opt, model, x)
        ff_dicts, target_inds = get_f_f_dict(opt, model_output, target=y, n_support=opt.K)
        ff_dicts = torch.reshape(ff_dicts, (ff_dicts.size(0), opt.C, opt.K))
        ft_dicts, test_dicts = get_f_t_dict(opt, model_output, text_features, target=y, n_support=opt.K)
        ft_dicts = torch.reshape(ft_dicts, (ft_dicts.size(0), opt.C, opt.K))

        ff_sorted_dist, _ = torch.sort(ff_dicts, dim=-1)
        ft_sorted_dist, _ = torch.sort(ft_dicts, dim=-1)

        batch_sizes = ft_dicts.size(0)
        C = opt.C
        K = opt.K
        all_clipped_mean_upper, all_clipped_mean_lower = [], []
        for t in range(T+1):
            clipped_mean_upper, clipped_mean_lower = Liptchz_bound(t, K, clip_k, ff_sorted_dist, ft_sorted_dist, test_dicts, opt.Lr, opt.Lambda)
            all_clipped_mean_upper.append(clipped_mean_upper)
            all_clipped_mean_lower.append(clipped_mean_lower)


        for t in range(T):
            all_preds = []
            #Suppose the attacker modifies t1 data samples in the ground-truth class, and t2 data samples in the non-ground-truth class
            #Consider all combinations of t1 and t2
            for t1 in range(t+1):
                t2 = t-t1
                bounds = torch.zeros(batch_sizes,C)
                for i in range(batch_sizes):
                    for j in range(C):
                        if torch.flatten(target_inds)[i] == j:
                            bounds[i][j] = all_clipped_mean_upper[t1][i][j]
                        else:
                            bounds[i][j] = all_clipped_mean_lower[t2][i][j]
                preds = torch.argmin(bounds, dim = -1)
                all_preds.append(preds)

            all_preds = torch.stack(all_preds, dim = 0)
            preds_poisoned = torch.zeros(all_preds.shape[1])
            all_preds = torch.transpose(all_preds, 0, 1)

            for i in range(all_preds.shape[0]):
                pred = all_preds[i]
                if torch.all(torch.eq(pred, pred[0])):
                    preds_poisoned[i] = pred[0]
                else:
                    preds_poisoned[i] = -1
            correct_count = 0

            for i in range(torch.flatten(target_inds).shape[0]):
                if preds_poisoned[i] == torch.flatten(target_inds)[i]:
                    correct_count+=1

            avg_acc[batch_id,t] = correct_count/torch.flatten(target_inds).shape[0]
    avg_acc = np.mean(avg_acc, axis=0)
    return avg_acc.tolist()


import torch
import numpy as np
from .fs_utils import *
import time
def certify_knn(opt, test_dataloader, model,n = 10,T =10,certify_type = "group",num_batches=20):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = opt.device
    #print(device)
    num_batches=test_dataloader.__len__()
    model = model.to(device)
    avg_acc = np.zeros((num_batches, T))
    count = 0

    test_iter = iter(test_dataloader)
    for batch in test_iter:
        count+=1
        if count == num_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_output = get_feature(opt,model,x)
        dist, target_inds = get_distances(opt,model_output, target=y,
                         n_support=opt.K)


        dist = torch.reshape(dist, (dist.size(0), opt.C,opt.K))
        chosen_distances, chosen_indices = torch.topk(dist.view((dist.size(0), opt.C*opt.K)), n,largest=False, dim=-1)

        counts = torch.zeros((dist.shape[0],dist.shape[1]))

        for i in range(dist.shape[0]):
            for cls_ind in range(dist.shape[1]):
                for sample_ind in range(dist.shape[2]):
                    if torch.any(torch.eq(chosen_distances[i], dist[i][cls_ind][sample_ind])):
                        counts[i][cls_ind]+=1
        for t in range(T):
            if certify_type == "ind":
                t = t*opt.C
            if certify_type == "group":
                t = t
            if t>=n/2:
                correct_count = 0
            else:
                topk_counts, topk_indices = torch.topk(counts, 2,dim=-1)

                topk_counts_bounds = topk_counts.clone()
                topk_counts_bounds[:,0] = topk_counts[:,0]-t
                topk_counts_bounds[:,1] = topk_counts[:,1]+t


                # for i in range(topk_indices.shape[0]):
                #     if topk_indices[i][0]<topk_indices[i][1]:
                #         topk_counts[i][0]+=0.001
                #         topk_counts_bounds[i][0]+=0.001
                #     else:
                #         topk_counts[i][0]-=0.001
                #         topk_counts_bounds[i][0]-=0.001


                orig_preds = torch.zeros(topk_indices.shape[0])
                argmax_orig = torch.argmax(topk_counts,dim = 1)
                poisoned_preds = torch.zeros(topk_indices.shape[0])
                argmax_poisoned = torch.argmax(topk_counts_bounds,dim = 1)

                for i in range(topk_indices.shape[0]):
                    orig_preds[i] = topk_indices[i][argmax_orig[i]]
                    poisoned_preds[i] = topk_indices[i][argmax_poisoned[i]]

                correct_count = 0
                for i in range(topk_indices.shape[0]):
                    if orig_preds[i] == poisoned_preds[i] and (orig_preds[i] == torch.flatten(target_inds)[i]):
                        correct_count+=1

            acc = correct_count/torch.flatten(target_inds).shape[0]

            avg_acc[count-1][t] = acc

    avg_acc = np.mean(avg_acc,axis=0)
  #  print('{},'.format(avg_acc))

    return avg_acc.tolist()



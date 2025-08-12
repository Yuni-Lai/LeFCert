import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from .fs_utils import *

def certify_dpa(opt, test_dataloader, model, n_partitions=5, T=1, certify_type="group",num_batches=20):
    model.eval()
    device = opt.device
    # print(device)
    model = model.to(device)
    avg_acc = np.zeros((num_batches, T))

    for batch_id, batch in enumerate(test_dataloader):
        if batch_id == num_batches:
            break
        features = get_feature(opt, model, batch, batch['node_id'].to(opt.device))
        batch_labels = batch['label'].to(opt.device)

        # Divide the support set and the query set
        current_classes = batch_labels.unique()[:opt.C]
        temp_labels = torch.zeros_like(batch_labels) #Create temporary labels (from 0 to C-1)
        for new_label, old_label in enumerate(current_classes):
            temp_labels[batch_labels == old_label] = new_label

        support_indices = []
        for cls in range(opt.C):
            cls_indices = (temp_labels == cls).nonzero().squeeze()
            support_indices.append(cls_indices[:opt.K])
        support_indices = torch.cat(support_indices)
        
        support_features = features[support_indices]
        support_labels = temp_labels[support_indices]
        query_mask = ~torch.isin(torch.arange(len(batch_labels)).to(device), support_indices).to(device)
        query_features = features[query_mask]
        query_labels = temp_labels[query_mask]

        # Partitioned training 
        # K samples of the support set were evenly distributed into C partitions (with K/C samples in each partition):
        partition_ids = (torch.arange(len(support_indices))) % n_partitions #hash
        partitions = []
        for i in range(n_partitions):
            part_mask = (partition_ids == i)
            part_indices = support_indices[part_mask]
            partitions.append((
                features[part_indices].cpu().numpy(),
                temp_labels[part_indices].cpu().numpy()
            ))
        classifiers = []
        for feats, labels in partitions:
            clf = LogisticRegression(max_iter=1000) #easiest way
            clf.fit(feats, labels)
            classifiers.append(clf)

        # vote and certify
        votes = torch.zeros((len(query_features), opt.C), device=device)
        for clf in classifiers:
            preds = clf.predict(query_features.cpu().numpy())
            for i, cls in enumerate(preds):
                if 0 <= cls < opt.C:
                    votes[i, cls] += 1

        top2_votes, _ = torch.topk(votes, 2, dim=1)

        for t in range(T):

            if certify_type == "ind":
                robust = (top2_votes[:, 0] - top2_votes[:, 1]) > t
            elif certify_type == "group":
                max_affected_votes = min(t, n_partitions)
                robust = (top2_votes[:, 0] - top2_votes[:, 1]) > 2 * max_affected_votes

            correct = (votes.argmax(dim=1) == query_labels).float()
            # avg_acc.append((correct * robust).mean().item())
            avg_acc[batch_id, t] = (correct * robust).mean().item()
    
    return np.mean(avg_acc,axis=0).tolist()

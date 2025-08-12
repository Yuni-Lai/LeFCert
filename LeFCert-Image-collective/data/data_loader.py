import torch
#from prototypical_batch_sampler import PrototypicalBatchSampler
from data import *
from utils import *
from torchvision import transforms
import numpy as np
import torch
import clip
from FCert import *
import json

def init_dataset(opt, mode):
    print("dataset mode:", mode)
    dataset_name = opt.dataset_type
    size = 224
    if dataset_name == "cifarfs":
        if opt.certify_model == 'LeFCert-LD':
            size = 32
            dataset = CIFARFS(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ]), download=True)
        else:
            dataset = CIFARFS(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4914, 0.4822, 0.4465]),
                                     np.array([0.2023, 0.1994, 0.2010]))
            ]), download=True)
        dataset.idx_to_class=reverse_dict(dataset.class_to_idx)
        labels = []
        for i in range(len(dataset)):
            labels.append(dataset[i][1])
        dataset.y = labels

    if dataset_name == "cubirds200":
        if opt.certify_model == 'LeFCert-LD':
            size = 256
            dataset = CUBirds200(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ]), download=True)
        else:
            dataset = CUBirds200(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ]), download=True)
        labels = []
        for i in range(len(dataset)):
            labels.append(dataset[i][1])
        dataset.y = labels

    if dataset_name == "tiered_imagenet":
        if opt.certify_model == 'LeFCert-LD':
            size = 256
            dataset = TieredImagenet(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ]), download=True)
        else:
            dataset = TieredImagenet(mode=mode, root='../FCert-Image/data', transform=transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ]), download=True)
        dataset.y = dataset.labels


    n_classes = len(np.unique(dataset.y))
    print("total n_classes:", n_classes)

    if n_classes < opt.C:
        raise (Exception('There are not enough classes in the dataset in order ' +
                         'to satisfy the chosen classes_per_it. Decrease the ' +
                         'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    classes_per_it = opt.C
    num_samples = opt.K + opt.num_query
    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataset,dataloader

def reverse_dict(my_dict):
    return {value: key for key, value in my_dict.items()}

import torch
#from prototypical_batch_sampler import PrototypicalBatchSampler
from torchvision import transforms
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
from models import GraphCLIP
import os
import pprint
import argparse
from data import *
from utils.utils import *
from FCert import *
from FCert.certifyKNN import *
from FCert.certifyDPA import *
from visualize import merged_certified_curve, plot_certified_accuracy
# Model Settings=======================================
parser = argparse.ArgumentParser(description='GraphClip Few-shot Learning Configuration')
parser.add_argument('--device', type=str, default='cuda:8', help='Device to use (cuda/cpu)')
parser.add_argument('--dataset_root',type=str,help='path to dataset',default='..' + os.sep + 'dataset')
parser.add_argument('--iterations',type=int,default=100,
                    help='number of episodes per epoch, the batch size is iterations/num_classes')
parser.add_argument( '--C',type=int,default=5,
                    help='number of random classes per episode for validation')
parser.add_argument( '--K',type=int,default=10,
                    help='number of samples per class to use as support for validation')
parser.add_argument('--num_query',type=int,default=1,
                    help='number of samples per class to use as query, default=1')
parser.add_argument('--manual_seed',type=int,default=7,
                    help='input for the manual seeds initializations')
parser.add_argument('--cuda',default=True,help='enables cuda')
parser.add_argument('--certify_type',type=str, help='group or individual',default="group", choices=['ind', 'group'])
parser.add_argument('--model_type',type=str,help='',default="GraphCLIP")
parser.add_argument('--certify_model', type=str, choices=['FCert','LeFCert','MLFCert',"KNN", "DPA"], default='LeFCert')
parser.add_argument( '--M',type=int,default=3,help='number of trimmed samples (K prime),default M = (K-1)/2')
parser.add_argument( '--Lambda',type=float,default=0.7,help='Lambda is a hyperparameter that controls the weight of the text feature in MFCert')
parser.add_argument('--box_constraint', action='store_true', default=False,
                    help="employ the box constraint of the cosine similarity [0,2],MFCert only, default False")
parser.add_argument('--dataset_type',type=str,help='',default='Citeseer',choices=['Cora','Citeseer','Computer','WikiCS','Photo'])
parser.add_argument( '--file_path',type=str,help='',default='./output/')
parser.add_argument('--use_subgraph', type=int, choices=[0, 1], default=1, help='1=True/0=False (default: 1)')


options = parser.parse_args()

if options.certify_model == 'LeFCert':
    options.metric_type = "cosine"
else:
    options.metric_type = "euclidean"

options.M = int((options.K-1)/2)# default

options.file_path = options.file_path+f'{options.dataset_type}_{options.model_type}/{options.certify_model}/C{options.C}_K{options.K}_M{options.M}'
if not os.path.exists(options.file_path):
    os.makedirs(options.file_path)

pprint.pprint(vars(options), width=1)

K = options.K
C = options.C
M = options.M #K'
CERTIFIY_TYPE = options.certify_type
MODEL_TYPE =  options.model_type
DATASET_TYPE = options.dataset_type
CERTIFY_MODEL = options.certify_model

init_seed(options)

if MODEL_TYPE == "GraphCLIP":
    attn_kwargs = {'dropout': 0.0}
    model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model="tiny")  # same as GraphCLIP setting
    ckpt_path = "./checkpoints/pretrained_graphclip.pt"
    model.load_state_dict(torch.load(ckpt_path),strict=False)  # only one parameter related to text(not important) is missing, so use strict=false, no harm
    model.to(options.device)
    model.eval()
else:
    raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented yet.")

dataset,test_dataloader = init_dataloader(options)
class_map=dataset.idx_to_class

result_dict = {
    "certify_model": CERTIFY_MODEL,
    "poisoning size": [i for i in range(10)],
    'certified accuracy': [],
    "K": K,
    "C": C,
    "M": M,
    "lambda":options.Lambda,
    "certify_type":CERTIFIY_TYPE,
    "model_type":MODEL_TYPE,
    "dataset_type":DATASET_TYPE
    }

with torch.no_grad():
    print(f"Attack type: {CERTIFIY_TYPE}, Model: {MODEL_TYPE}, Dataset: {DATASET_TYPE}, K={K}, C={C}, M={M}")
    print(f"Certify model: {CERTIFY_MODEL}, box_constraint: {options.box_constraint}, Lambda:{options.Lambda} (MFCert only)")
    
    if options.certify_model == "FCert":
        acc = FCert(opt=options,test_dataloader=test_dataloader,
                model=model,clip_k=M,T =10,certify_type=CERTIFIY_TYPE)
    elif options.certify_model == 'LeFCert':
        acc = MFCert(opt=options,test_dataloader=test_dataloader,
                model=model,clip_k=M,T =10,certify_type=CERTIFIY_TYPE,class_map=class_map)
    elif options.certify_model == "MLFCert":
        acc = MLFCert(opt=options,test_dataloader=test_dataloader,
                model=model,clip_k=M,T =10,certify_type=CERTIFIY_TYPE,class_map=class_map)
    elif options.certify_model == "KNN":
        acc = certify_knn(opt=options, test_dataloader=test_dataloader, 
                model=model, n=10, T=10, certify_type="group" )
    elif options.certify_model == "DPA":
        acc = certify_dpa( opt=options, test_dataloader=test_dataloader, 
                model=model, n_partitions=K, T=10
        )
    #result_dict['certified accuracy'].append(acc)
'''
df = pd.DataFrame(result_dict)
f = open(f'{options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl')
print(df.to_string())
'''
result_dict['certified accuracy']=acc
df = pd.DataFrame(result_dict)
f = open(f'{options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl')
print(df.to_string())



if options.certify_model in ['LeFCert']:
    dfs = []
    for model_name in ['FCert', 'LeFCert', 'KNN', 'DPA']:
        try:
            out_dir = f'./output/{options.dataset_type}_{options.model_type}/{model_name}/C{options.C}_K{options.K}_M{options.M}'
            f = open(f'{out_dir}/certify_result_{CERTIFIY_TYPE}.pkl', 'rb')
            df = pickle.load(f)
            dfs.append(df)
            f.close()
        except FileNotFoundError:
            print(f"Warning: {model_name} result not found, skipping.")
    if dfs:
        options.file_path = f'./output/{options.dataset_type}_{options.model_type}/C{options.C}_K{options.K}_M{options.M}'
        df_combine = pd.concat(dfs, ignore_index=True)
        merged_certified_curve(df_combine, options.file_path, options)

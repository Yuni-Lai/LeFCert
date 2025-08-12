import os
import argparse
import pandas as pd
import pickle
import torch
from data import *
from utils import *
from torchvision import transforms
import numpy as np
from FCert.certifyKNN import certify_knn
from FCert.certifyDPA import certify_dpa
import clip
from FCert import FCert, LeFCert,LeFCertL, LeFCertNew
from data.data_loader import init_dataset, init_sampler, init_dataloader,reverse_dict
from visualize import merged_certified_curve, name_map
from Diffusion.DRM import DiffusionRobustModel
from Diffusion.Smoothing import Smooth
# Model Settings=======================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root',type=str,help='path to dataset',default='..' + os.sep + 'dataset')
parser.add_argument('--iterations',type=int,default=10,
                    help='number of batches, the batch size is C')
parser.add_argument( '--C',type=int,default=5,
                    help='number of random classes per episode for validation')
parser.add_argument( '--K',type=int,default=25,
                    help='number of samples per class to use as support for validation')
parser.add_argument('--num_query',type=int,default=1,
                    help='number of samples per class to use as query, default=1')
parser.add_argument('--manual_seed',type=int,default=7,
                    help='input for the manual seeds initializations')
parser.add_argument('--cuda',default=True,help='enables cuda')
parser.add_argument('--certify_type',type=str, help='group or individual',default="group")
parser.add_argument('--model_type',type=str,help='the base foundation model',default="CLIP")
parser.add_argument('--certify_model', type=str, choices=['KNN','DPA','FCert','LeFCert','LeFCert-L','LeFCert-LD'], default="LeFCert")
parser.add_argument('--metric_type',type=str,default="cosine",choices=['cosine','l2'])
parser.add_argument( '--M',type=int,default=3,help='number of trimmed samples (K prime),default M = (K-1)/2')
parser.add_argument( '--Lambda',type=float,default=10,help='Lambda is a hyperparameter that controls the weight of the text feature in LeFCert')
parser.add_argument('--box_constraint', action='store_true', default=False,
                    help="employ the box constraint of the cosine similarity [0,2]")
parser.add_argument('--sigma', type=float, default=1.0, help='sigma for LeFCertL, the noise level for smoothing')
parser.add_argument('--Lr', type=float, help='Lr=Lipschitz constant for the model x attack radius')
parser.add_argument('--dataset_type',type=str,help='',default="tiered_imagenet",choices=['cifarfs','cubirds200','tiered_imagenet'])
parser.add_argument( '--file_path',type=str,help='',default='./output/')
options = parser.parse_args()

if options.certify_model in ['LeFCert-L', 'LeFCert-LD']:
    options.metric_type = "l2"  # LeFCertL only supports l2 metric
    options.r=0.25 # default radius for the attacker constraint
    r = options.r
    options.Lr = np.sqrt(2 / (np.pi * options.sigma ** 2)) * r

if options.metric_type == "l2":
    options.Lambda = 0.4 #"l2"
else:
    options.Lambda = 25 #"cosine"

options.M = int((options.K-1)/2)# default

options.file_path = options.file_path+f'{options.dataset_type}_{options.model_type}/{options.certify_model}/C{options.C}_K{options.K}_M{options.M}'
if not os.path.exists(options.file_path):
    os.makedirs(options.file_path)

K = options.K
C = options.C
M = options.M #K'
CERTIFIY_TYPE = options.certify_type
MODEL_TYPE =  options.model_type
DATASET_TYPE = options.dataset_type
CERTIFY_MODEL = options.certify_model

if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
options.device = 'cuda:1' if torch.cuda.is_available() and options.cuda else 'cpu'


init_seed(options)
dataset,test_dataloader = init_dataloader(options, 'test')
class_map=dataset.idx_to_class
if MODEL_TYPE == "CLIP":
    model, preprocess = clip.load('ViT-B/32', 'cuda')
else:
    print("Invalid model type")
# Initialize the saved dictionary

if options.certify_model in ['LeFCert-L', 'LeFCert-LD']:
    diff_model = DiffusionRobustModel(dataset=DATASET_TYPE)
    # Get the timestep t corresponding to noise level sigma
    target_sigma = options.sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = diff_model.diffusion.sqrt_alphas_cumprod[t]
        b = diff_model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a
    # Define the smoothed classifier
    smoothed_classifier = Smooth(model,diff_model, options.sigma, t)


Max_T=10
if K>10:
    Max_T = 15
result_dict = {
"certify_model": CERTIFY_MODEL,
"poisoning size": [i for i in range(Max_T)],
'certified accuracy': [],
"K": K,
"C": C,
"M": M,
"metric_type": options.metric_type,
"certify_type":CERTIFIY_TYPE,
"model_type":MODEL_TYPE,
"dataset_type":name_map[DATASET_TYPE]
}


with torch.no_grad():
    print(f"Attack type: {CERTIFIY_TYPE}, Model: {MODEL_TYPE}, Dataset: {DATASET_TYPE}, K={K}, C={C}, M={M}")
    print(f"Certify model: {CERTIFY_MODEL}, box_constraint: {options.box_constraint}, Lambda:{options.Lambda} (for LeFCert or LeFCertL)")
    print(f"Using device: {options.device}")
    if options.certify_model == "FCert":
        acc = FCert(opt=options,test_dataloader=test_dataloader,
              model=model,clip_k=M,T=Max_T,certify_type = CERTIFIY_TYPE)
    elif options.certify_model == 'LeFCert':
        acc = LeFCert(opt=options,test_dataloader=test_dataloader,
                model=model,clip_k=M,T=Max_T,certify_type = CERTIFIY_TYPE,class_map=class_map)
    elif options.certify_model == 'LeFCert-new':
        acc = LeFCertNew(opt=options, test_dataloader=test_dataloader,
                     model=model, clip_k=M, T=Max_T, certify_type=CERTIFIY_TYPE, class_map=class_map)
    elif options.certify_model == 'LeFCert-L':
        acc = LeFCertL(opt=options,test_dataloader=test_dataloader,
                model=smoothed_classifier,clip_k=M,T=Max_T,certify_type = CERTIFIY_TYPE,class_map=class_map)
    elif options.certify_model == 'LeFCert-LD':
        acc = LeFCertL(opt=options,test_dataloader=test_dataloader,
                model=smoothed_classifier,clip_k=M,T=Max_T,certify_type = CERTIFIY_TYPE,class_map=class_map)
    elif options.certify_model == "KNN":
        acc = certify_knn(opt=options, test_dataloader=test_dataloader,
                model=model, n=10, T=Max_T, certify_type=CERTIFIY_TYPE )
    elif options.certify_model == "DPA":
        acc = certify_dpa( opt=options, test_dataloader=test_dataloader,
                model=model, n_partitions=K, T=Max_T,certify_type = CERTIFIY_TYPE)
result_dict['certified accuracy']=acc
if options.certify_model in ['LeFCert-L', 'LeFCert-LD']:
    result_dict['sigma'] = options.sigma
    result_dict['Lambda'] = options.Lambda
    result_dict['r'] = r
    CERTIFIY_TYPE = f"{CERTIFIY_TYPE}_sigma{options.sigma}_r{r}_ld{options.Lambda}"
elif options.certify_model in ['LeFCert', 'LeFCert-new']:
    result_dict['Lambda'] = options.Lambda
    CERTIFIY_TYPE = f"{CERTIFIY_TYPE}_ld{options.Lambda}"

df = pd.DataFrame(result_dict)
f = open(f'{options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl', 'wb')
pickle.dump(df, f)
f.close()
print(f'Save result to {options.file_path}/certify_result_{CERTIFIY_TYPE}.pkl')
print(df.to_string())

if options.certify_model in ['LeFCert-L','LeFCert-LD']:
    dfs = []
    for model_name in ['LeFCert-LD','LeFCert-L','LeFCert','FCert', 'KNN', 'DPA']:
        try:
            if model_name in ['LeFCert-L', 'LeFCert-LD']:
                CERTIFIY_TYPE = f"{options.certify_type}_sigma{options.sigma}_r{r}_ld{options.Lambda}"
            elif model_name == 'LeFCert':
                CERTIFIY_TYPE = f"{options.certify_type}_ld{options.Lambda}"
            else:
                CERTIFIY_TYPE = options.certify_type
            out_dir = f'./output/{options.dataset_type}_{options.model_type}/{model_name}/C{options.C}_K{options.K}_M{options.M}'
            f = open(f'{out_dir}/certify_result_{CERTIFIY_TYPE}.pkl', 'rb')
            df = pickle.load(f)
            dfs.append(df)
            f.close()
        except FileNotFoundError:
            print(f"Warning: {model_name} result not found, skipping.")
    
    if dfs:
        options.file_path = f'./output/{options.dataset_type}_{options.model_type}/merged_certaccuracy_C{options.C}_K{options.K}_M{options.M}_sig{options.sigma}_r{r}.pdf'
        df_combine = pd.concat(dfs, ignore_index=True)
        merged_certified_curve(df_combine, options.file_path, options)
    part_of_df=df_combine[df_combine.loc[:, "poisoning size"].isin([0, 1, 3, 5, 7, 9])]
    print(f'Dataset:{DATASET_TYPE},C{options.C}_K{options.K}_M{options.M}_sig{options.sigma}_r{r}')
    print(part_of_df.pivot_table(index='poisoning size', columns='certify_model', values='certified accuracy'))

if options.certify_model in ['LeFCert-new']:
    dfs = []
    for model_name in ['LeFCert', 'LeFCert-new']:
        try:
            CERTIFIY_TYPE = f"{options.certify_type}_ld{options.Lambda}"
            out_dir = f'./output/{options.dataset_type}_{options.model_type}/{model_name}/C{options.C}_K{options.K}_M{options.M}'
            f = open(f'{out_dir}/certify_result_{CERTIFIY_TYPE}.pkl', 'rb')
            df = pickle.load(f)
            dfs.append(df)
            f.close()
        except FileNotFoundError:
            print(f"Warning: {model_name} result not found, skipping.")

    if dfs:
        options.file_path = f'./output/{options.dataset_type}_{options.model_type}/merged_certaccuracy_C{options.C}_K{options.K}_M{options.M}.pdf'
        df_combine = pd.concat(dfs, ignore_index=True)
        merged_certified_curve(df_combine, options.file_path, options)
    part_of_df = df_combine[df_combine.loc[:, "poisoning size"].isin([0, 1, 2, 3, 4, 5])]
    print(f'Dataset:{DATASET_TYPE},C{options.C}_K{options.K}_M{options.M}')
    print(part_of_df.pivot_table(index='poisoning size', columns='certify_model', values='certified accuracy'))
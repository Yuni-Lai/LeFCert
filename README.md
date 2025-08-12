# Provably Robust Adaptation for Language-Empowered Foundation Models




## Introduction
This is the official PyTorch implementation of [**Provably Robust Adaptation for Language-Empowered Foundation Models**](https://arxiv.org/abs/2404.08631).
**LeFCert** is a language-empowered few-shot certification. 

This code is adapted from:
```
@article{wang2024fcert,
  title={FCert: Certifiably Robust Few-Shot Classification in the Era of Foundation Models},
  author={Wang, Yanting and Zou, Wei and Jia, Jinyuan},
  journal={arXiv preprint arXiv:2404.08631},
  year={2024}
}
```


## Environment
In this repo, we implement LeFCert for [CLIP](https://github.com/openai/CLIP) on three image datasets and [GraphCLIP](https://github.com/zhuyun97/graphclip) on two graph datasets. We test our code in Python 3.7, CUDA 11.8, and PyTorch 1.13.1.

```bash
cd ./Environments
conda env create -f LeFCert.yml
conda activate LeFCert
pip install -r LeFCert.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.


Please install [CLIP](https://github.com/openai/CLIP) by:
```
pip install git+https://github.com/openai/CLIP.git
```

For GraphCLIP, the environment is different, and please refer to ./FCert-GraphCLIP/README.md for details.

## Usage

To evaluate the certification performance of LeFCert:
```
cd ./LeFCert-Image
python -u main.py --dataset_type cifarfs --certify_model 'LeFCert' --C 5 --K 10
```

To evaluate the variants of LeFCert-LD:
```
cd ./LeFCert-Image
python -u main.py --dataset_type cifarfs --certify_model 'LeFCert-LD' --C 5 --K 10
```
To evaluate the variants of LeFCert-C:
```
cd ./LeFCert-Image-collective
python -u main.py --dataset_type cifarfs --certify_model 'LeFCert-C' --C 5 --K 10
```

To evaluate the certification performance of LeFCert on graph datasets:
```cd ./LeFCert-GraphCLIP
python -u main.py --dataset_type 'cora' --certify_model 'LeFCert' --C 5 --K 10
```

## License
The paper is under review. Please do not distribute this code without the authors' permission. 




# LeFCert for GraphCLIP

**LeFCert** is a language-empowered few-shot certification. 


## Set up
We use the GraphCLIP model, the set up of pre-trained GraphCLIP can be found here:https://github.com/zhuyun97/graphclip.
Download the [released checkpoint](https://drive.google.com/file/d/178RikDLXPy-4eMGDhG5V6RzmlJhp-8fy/view?usp=sharing), unzip it and put the extracted `pretrained_graphclip.pt` in ./checkpoints.

We test our code in Python 3.9, CUDA 11.8, and PyTorch 2.7.0, torch-geometric 2.6.1:
```bash
cd ./Environments
conda env create -f py39.yml
conda activate py39
pip install -r py39.txt
```
if report: "ResolvePackageNotFound:xxx", or "No matching distribution found for xxx", just open the .yaml or .txt file and delete that line.





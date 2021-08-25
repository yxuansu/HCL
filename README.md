# Dialogue Response Selection with Hierarchical Curriculum Learning
Authors: Yixuan Su, Deng Cai, Qingyu Zhou, Zibo Lin, Simon Baker, Yunbo Cao, Shuming Shi, Nigel Collier, and Yan Wang

## Introduction:
In this repository, we provide a simpler and more robust implementation of our paper [link](https://aclanthology.org/2021.acl-long.137.pdf). We provide data, pre-trained models for Douban dataset. We will update data for Ubuntu and E-commerce soon. 

## 1. Enviornment Installtion:
```yaml
pip install -r requirements.txt
```
```yaml
```

## 2. Download Data [here](https://drive.google.com/file/d/13Fzd91hcJ84abv6RwOKmhInSK0yxQxTx/view?usp=sharing):
```yaml
unzip data.zip and replace it with the empty ./data folder.
```

## 3. SABERT
### (1) GPU Requirement:
```yaml
a. 4 x Tesla V100 GPUs(16GB)
b. Cuda Version: 11.0
```
### (2) Download pre-trained BERT parameter [here](https://drive.google.com/file/d/1SECNJGgrBVewSRfTCUlXe_uEhXdyLhd9/view?usp=sharing):
```yaml
unzip bert-base-chinese.zip and replace it with the empty ./SABERT/bert-base-chinese folder
```
### (3) Training from scratch:
```yaml
cd ./SABERT
chmod +x ./train.sh
./train.sh
```
### (4) Inference from pre-trained checkpoints:
#### (a) Download pre-trained parameters [here](https://drive.google.com/file/d/1_lEXE4RpG67FOEE0V0Aj7_B1lEADuJ5u/view?usp=sharing):
```yaml
unzip ckpt.zip and replace it with the empty ./SABERT/ckpt folder
cd ./SABERT
chmod +x ./inference.sh
./inference.sh
```









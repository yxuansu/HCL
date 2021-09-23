# Dialogue Response Selection with Hierarchical Curriculum Learning
Authors: Yixuan Su, Deng Cai, Qingyu Zhou, Zibo Lin, Simon Baker, Yunbo Cao, Shuming Shi, Nigel Collier, and Yan Wang

Code for ACL 2021 paper [Dialogue Response Selection with Hierarchical Curriculum Learning](https://aclanthology.org/2021.acl-long.137.pdf)

## Introduction:
In this repository, we provide a simpler and more robust implementation of our ACL 2021 paper and it requires less hyper-parameter tuning. We provide data, pre-trained models for Douban dataset. We will update data for Ubuntu and E-commerce soon. 

## 1. Enviornment Installtion:
```yaml
pip install -r requirements.txt
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
```
#### (b) Perform inference:
```yaml
cd ./SABERT
chmod +x ./inference.sh
./inference.sh
```

## 4. SMN and MSN
### (1) GPU Requirement:
```yaml
a. 1 x Tesla V100 GPUs(16GB)
b. Cuda Version: 11.0
```
### (2) Download embeddings for both models [here](https://drive.google.com/file/d/1jFrIdP-CyrSjklqSXmH2sNbA-on7jgNP/view?usp=sharing):
```yaml
unzip embeddings.zip and replace it with the empty ./SMN_MSN/embeddings folder
```
### (3) Training from scratch:
```yaml
cd ./SMN_MSN/
chmod +x train_X.sh (X in ['smn', 'msn])
./train_X.sh
```
### (4) Inference from pre-trained checkpoints:
#### (a) Download pre-trained parameters [here](https://drive.google.com/file/d/1xrCEeTNtHLqRfy35fE7cKHXTkzTIPPXM/view?usp=sharing):
```yaml
unzip ckpt.zip and replace it with the empty ./SMN_MSN/ckpt folder
```
#### (b) Perform inference:
```yaml
cd ./SMN_MSN
chmod +x ./inference_X.sh (X in ['smn', 'msn])
./inference_X.sh
```

## 5.Citation
If you find our paper and resources useful, please kindly cite our paper:

    @inproceedings{su-etal-2021-dialogue,
        title = "Dialogue Response Selection with Hierarchical Curriculum Learning",
        author = "Su, Yixuan  and
          Cai, Deng  and
          Zhou, Qingyu  and
          Lin, Zibo  and
          Baker, Simon  and
          Cao, Yunbo  and
          Shi, Shuming  and
          Collier, Nigel  and
          Wang, Yan",
        booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
        month = aug,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.acl-long.137",
        doi = "10.18653/v1/2021.acl-long.137",
        pages = "1740--1751"
    }











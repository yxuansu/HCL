# -*- coding:utf-8 -*-
import torch
import pickle
import collections
import time
import numpy as np
import os

def get_model_name(root_dir, prefix):
    for filename in os.listdir(root_dir):
        if filename.startswith(prefix):
            model_name = filename
            break
    return model_name
# ------------------------------------------------------------------------------------- #
# measurement functions
def compute_Rn_k(scores,labels, n=2, k=1):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = np.asarray(scores[i:i+n])
            index = sublist.argsort()[::-1][0:k]
            if scores[i] in sublist[index]:
                correct = correct + 1
    return float(correct) / total

def compute_R2_1(scores,labels, n=10, k=1, m=2):
    total = 0
    correct = 0
    for i in range(0, len(labels), n):
        total = total+1
        true_response_index = []
        for j in range(i, i+n):
            if labels[j] == 1:
                true_response_index.append(j-i)
        sublist = np.asarray(scores[i:i+m])
        index = sublist.argsort()[::-1][0:k]
        # if len(np.intersect1d(index, true_response_index)) > 0:
        #         correct = correct + 1
        correct += len(np.intersect1d(index, true_response_index)) * 1.0 / len(true_response_index)
    return float(correct) / total

def compute_R10_k(scores,labels, n=10, k=1):
    total = 0
    correct = 0
    for i in range(0, len(labels), n):
        total = total+1
        true_response_index = []
        for j in range(i, i+n):
            if labels[j] == 1:
                true_response_index.append(j-i)
        sublist = np.asarray(scores[i:i+n])
        index = sublist.argsort()[::-1][0:k]
        # if len(np.intersect1d(index, true_response_index)) > 0:
        #         correct = correct + 1
        correct += len(np.intersect1d(index, true_response_index)) * 1.0 / len(true_response_index)
    return float(correct) / total

def compute_P1(scores, labels, n=10):
    '''precision at position 1'''
    total = 0
    correct = 0
    for i in range(0, len(labels), n):
        total = total+1
        sublist = np.asarray(scores[i:i+n])
        index = sublist.argsort()[::-1]
        p1 = 0.0
        if labels[i+index[0]] == 1: p1 = 1.0
        correct += p1
    return float(correct) / total

def compute_MAP(scores,labels, n=10):
    total = 0
    correct = 0
    for i in range(0, len(labels), n):
        total = total+1
        sublist = np.asarray(scores[i:i+n])
        index = sublist.argsort()[::-1]
        ap = 0.0
        count = 0
        for j, ans_index in enumerate(index):
            if labels[i+ans_index] == 1:
                count+=1
                ap += count / (j+1.0)
        correct += (ap / count)
    return float(correct) / total

def compute_MRR(scores,labels, n=10):
    total = 0
    correct = 0
    for i in range(0, len(labels), n):
        total = total+1
        sublist = np.asarray(scores[i:i+n])
        index = sublist.argsort()[::-1]
        ap = 0.0
        for j, ans_index in enumerate(index):
            if labels[i+ans_index] == 1:
                ap += 1.0 / (j+1.0)
                break
        correct += ap
    return float(correct) / total
        

# -*- coding:utf-8 -*-
import pickle
import collections
import progressbar
import time
import numpy as np
import os
UNK, PAD = '[UNK]', '[PAD]'

# loading response index
def load_response_id_dict(in_f):
    id_response_dict = {}
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        print ('Size of response index is %d' % len(lines))
        for one_id in range(len(lines)):
            one_response = lines[one_id].strip('\n')
            id_response_dict[one_id] = one_response
    print ('Size of response index is %d' % len(id_response_dict))
    return id_response_dict

# loading context data
def process_text(text, max_uttr_len, tokenizer):
    #token_list = text.strip().split()
    token_list = tokenizer.tokenize(text.strip())[:max_uttr_len]
    len_diff = max_uttr_len - len(token_list)
    token_list = token_list + [PAD for _ in range(len_diff)]
    assert len(token_list) == max_uttr_len
    return token_list

def process_context_text(context_text, max_uttr_num, max_uttr_len, tokenizer):
    padding_sen = [PAD for _ in range(max_uttr_len)]
    context_text_list = context_text.strip().split('\t')
    context_text_list = context_text_list[-max_uttr_num:]
    res_list = []
    for c_text in context_text_list:
        one_token_list = process_text(c_text.strip(), max_uttr_len, tokenizer)
        res_list.append(one_token_list)
    len_diff = max_uttr_num - len(res_list)
    for _ in range(len_diff):
        res_list.append(padding_sen)
    assert len(res_list) == max_uttr_num
    assert len(res_list[0]) == max_uttr_len
    res_id_list = []
    for one_token_list in res_list:
        res_id_list.append(tokenizer.convert_tokens_to_ids(one_token_list))
    return res_id_list # max_uttr_num x max_uttr_len

def load_context_data(in_f, max_uttr_num, max_uttr_len, tokenizer):
    context_id_list = []
    with open(in_f, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        data_idx = 0
        print ('Loading Context Data...')
        p = progressbar.ProgressBar(len(lines))
        p.start()
        for l in lines:
            #if data_idx % int(len(lines) / 5) == 0:
            #    print ('%d contexts have been loaded' % data_idx)
            one_context_text = l.strip('\n').strip()
            one_context_id = process_context_text(one_context_text, max_uttr_num, max_uttr_len, tokenizer)
            context_id_list.append(one_context_id)
            p.update(data_idx+1)
            data_idx += 1
        p.finish()
    return context_id_list

# loading dev data
def load_dev_data(path, max_uttr_num, max_uttr_len, tokenizer):
    '''
        each response candidate list contains 10 responses
        each candidate response label list contains 10 labels
    '''
    context_id_list, candi_response_id_list, candi_response_label_list = [], [], []
    with open(path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        test_data_num = int(len(lines) / 10)
        print ('test data number is %d' % test_data_num)
        print ('Loading Test Data...')
        p = progressbar.ProgressBar(len(lines))
        p.start()
        for i in range(test_data_num):
            #if i % int(test_data_num / 10) == 0:
            #    print ('%d test instances have been loaded' % i)
            p.update(i+1)
            batch_text_list = lines[i*10:(i+1)*10]
            batch_text_list = [text.strip('\n') for text in batch_text_list]
            one_context_text_list = batch_text_list[0].strip('\n').split('\t')[1:-1]
            one_context_text = '\t'.join(one_context_text_list).strip('\t')
            one_context_id = process_context_text(one_context_text, max_uttr_num, max_uttr_len, tokenizer)
            context_id_list.append(one_context_id)
            # process candidate response
            start_idx = i*10
            one_candi_response_list = []
            one_candi_response_label_list = []
            for candi_idx in range(start_idx, start_idx + 10):
                one_line = lines[candi_idx]
                one_line_content_list = one_line.strip('\n').split('\t')
                one_candi_response_label = int(one_line_content_list[0])
                one_candi_response_label_list.append(one_candi_response_label)
                one_candi_response_text = one_line_content_list[-1]
                one_candi_response_token_list = process_text(one_candi_response_text, max_uttr_len, tokenizer)
                one_candi_response_id_list = tokenizer.convert_tokens_to_ids(one_candi_response_token_list)
                one_candi_response_list.append(one_candi_response_id_list)
            candi_response_id_list.append(one_candi_response_list)
            candi_response_label_list.append(one_candi_response_label_list)
        p.finish()
        print ('Test Data Loaded.')
    return context_id_list, candi_response_id_list, candi_response_label_list

import os
import sys
import torch
import argparse
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from dataclass import Data, Tokenizer
import operator
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
from main_utlis import compute_Rn_k, compute_R2_1, compute_R10_k, compute_P1, compute_MAP, compute_MRR, read_pkl_file
import progressbar

def get_model_name(root_dir, prefix):
    for filename in os.listdir(root_dir):
        if filename.startswith(prefix):
            model_name = filename
            break
    return model_name


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--train_context_path', type=str)
    parser.add_argument('--train_true_response_path', type=str)
    parser.add_argument('--response_index_path', type=str)
    parser.add_argument('--train_context_vec_file', type=str)
    parser.add_argument('--all_response_vec_file', type=str)
    parser.add_argument('--negative_num', type=int, default=5)
    parser.add_argument('--dev_path', type=str)
    parser.add_argument('--word2id_path', type=str)
    parser.add_argument('--max_uttr_num', type=int, default=10)
    parser.add_argument('--max_uttr_len', type=int, default=50)
    # model configuration
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--embedding_path', type=str)
    # learning configuration
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str)
    return parser.parse_args()

def load_matching_model(args):
    embedding_matrix = read_pkl_file(args.embedding_path, "bytes")
    assert len(embedding_matrix) == Tokenizer(args.word2id_path).vocab_size
    padding_idx = Tokenizer(args.word2id_path).padding_idx
    if args.model_type == 'SMN':
        print ('Constructing new SMN model...')
        from modules.smn import SMN
        model = SMN(embedding_matrix, match_type=0, max_num_utterances=args.max_uttr_num)
    elif args.model_type == 'MSN':
        print ('Constructing new MSN model...')
        from modules.msn import MSN
        model = MSN(embedding_matrix, gru_hidden=300, padding_idx=padding_idx)
    else:
        raise Exception('Wrong Model Type!!!')
    return model

if __name__ == "__main__":
    args = parse_config()
    device = args.gpu_id

    print ('Load the best checkpoints from pretraining...')
    ckpt_name = get_model_name(args.ckpt_path, prefix = 'epoch')
    model_ckpt = torch.load(args.ckpt_path + '/' + ckpt_name)
    model_parameters = model_ckpt['model']

    model = load_matching_model(args)
    model.load_state_dict(model_parameters)
    model = model.cuda(device)
    print ('Model loaded.')

    negative_mode, mips_config = 'random_search', {}
    print ('Loading Data...')
    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path, args.negative_num, 
        args.dev_path, args.word2id_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode,
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    batch_size, dev_data_num = args.batch_size, data.dev_num
    dev_step_num = int(dev_data_num / batch_size) + 1
    model.eval()
    dev_score_list, dev_label_list = [], []
    print ('Start evaluation...')
    p = progressbar.ProgressBar(dev_step_num)
    p.start()
    with torch.no_grad():
        for dev_step in range(dev_step_num):
            p.update(dev_step)
            dev_batch_context_list, dev_batch_candidate_response_list, \
            dev_batch_candidate_response_label_list = data.get_next_dev_batch(batch_size)
            dev_batch_context_inp = torch.LongTensor(dev_batch_context_list).cuda(device)
            dev_batch_candidate_response_inp = torch.LongTensor(dev_batch_candidate_response_list).cuda(device)
            _, candi_num, _ = dev_batch_candidate_response_inp.size()
            dev_batch_score = model.batch_forward(dev_batch_context_inp, dev_batch_candidate_response_inp)
            assert dev_batch_score.size() == torch.Size([batch_size, candi_num])
            dev_batch_score = dev_batch_score.detach().cpu().tolist()
            dev_score_list += dev_batch_score
            dev_label_list += dev_batch_candidate_response_label_list
        p.finish()

        valid_dev_score_list = []
        for item in dev_score_list[:data.dev_num]:
            valid_dev_score_list.extend(item)
        valid_dev_label_list = []
        for item in dev_label_list[:data.dev_num]:
            valid_dev_label_list.extend(item)
        assert len(valid_dev_score_list) == len(valid_dev_label_list)

        MAP = compute_MAP(valid_dev_score_list, valid_dev_label_list, n = candi_num)
        MRR = compute_MRR(valid_dev_score_list, valid_dev_label_list, n = candi_num)
        P_1 = compute_P1(valid_dev_score_list, valid_dev_label_list, n = candi_num)
        R10_1 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 1)
        R10_2 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 2)
        R10_5 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 5)
        print ('----------------------------------------------------------------')
        print ('Test MAP:{}, MRR:{}, P@1:{}, R10@1:{}, R10@2:{}, R10@5:{}'.format(round(MAP, 4),
            round(MRR, 4), round(P_1, 4), round(R10_1, 4), round(R10_2, 4), round(R10_5, 4)))


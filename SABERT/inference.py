import os
import sys
import torch
import argparse
from torch import nn
import numpy as np
import torch.nn.functional as F
import random
from dataloader import Data
import operator
from operator import itemgetter
from transformers import AdamW, get_linear_schedule_with_warmup
from utlis import compute_Rn_k, compute_R2_1, compute_R10_k, compute_P1, compute_MAP, compute_MRR
from pytorch_pretrained_bert.modeling import BertModel
from model import Model
import progressbar

def get_model_name(root_dir, prefix):
    for filename in os.listdir(root_dir):
        if filename.startswith(prefix):
            model_name = filename
            break
    return model_name

def transform_input(list_of_batch_inp, device=None, use_cuda=False):
    res_list = []
    for item in list_of_batch_inp:
        one_res = torch.LongTensor(item)
        if use_cuda:
            one_res = one_res.to(device)
        res_list.append(one_res)
    return res_list

def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument('--corpus_name', type=str, help="The name of concerned corpus.")
    parser.add_argument('--train_context_path', type=str, help="The file contains all training context.")
    parser.add_argument('--train_true_response_path', type=str, help="The file contains all reference response.")
    parser.add_argument('--response_index_path', type=str, help="The file contains all responses in the dataset.")
    parser.add_argument('--train_context_vec_file', type=str, help="File contains context representations.")
    parser.add_argument('--all_response_vec_file', type=str, help="File contains response representations.")
    parser.add_argument('--dev_path', type=str, help="Validation data path.")
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--max_uttr_num', type=int, default=8, help="Maximum number of utterances in the context.")
    parser.add_argument('--max_uttr_len', type=int, default=30, help="Maximum length of each utterance.")
    parser.add_argument('--negative_num', type=int, default=5, help="Number of negative samples during training.")
    # learning configuration
    #parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument("--batch_size", type=int, help='Inference batch size.')  
    parser.add_argument('--ckpt_path', type=str, help="Path to save the model checkpoints.")
    return parser.parse_args()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()

    args = parse_config()
    device = torch.device('cuda')

    print ('Loading Data...')
    negative_mode, mips_config = 'random_search', {}
    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path,
        args.dev_path, args.bert_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode, 
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    print ('Load the best checkpoints...')
    ckpt_name = get_model_name(args.ckpt_path, prefix = 'epoch')
    model_ckpt = torch.load(args.ckpt_path + '/' + ckpt_name)
    model_parameters = model_ckpt['model']

    bert_model = BertModel.from_pretrained(args.bert_path)
    padding_idx = data.text_processor.padding_idx
    model = Model(bert_model, padding_idx)
    model.load_state_dict(model_parameters)
    if cuda_available:
        model = model.to(device) 
    print ('Pretrained checkpoints loaded.')

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
            dev_list_of_batch_token_id_inp, dev_list_of_batch_speaker_seg_id_inp, dev_list_of_batch_uttr_seg_id_inp, \
            dev_batch_candidate_response_label_list = data.get_next_dev_batch(batch_size)
            dev_list_of_batch_token_id_inp = transform_input(dev_list_of_batch_token_id_inp, device=device, use_cuda=True)
            dev_list_of_batch_speaker_seg_id_inp = transform_input(dev_list_of_batch_speaker_seg_id_inp, device=device, use_cuda=True)
            dev_list_of_batch_uttr_seg_id_inp = transform_input(dev_list_of_batch_uttr_seg_id_inp, device=device, use_cuda=True)
            dev_batch_score = model.batch_forward(dev_list_of_batch_token_id_inp, dev_list_of_batch_speaker_seg_id_inp, 
                                                          dev_list_of_batch_uttr_seg_id_inp, is_training = False)
            candi_num = len(dev_list_of_batch_token_id_inp)
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
        R2_1 = compute_R2_1(valid_dev_score_list, valid_dev_label_list, n = candi_num)
        R10_1 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 1)
        R10_2 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 2)
        R10_5 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 5)
        if args.corpus_name == 'douban':
            print ('Test MAP:{}, MRR:{}, P@1:{}, R10@1:{}, R10@2:{}, R10@5:{}'.format(round(MAP, 4),
                round(MRR, 4), round(P_1, 4), round(R10_1, 4), round(R10_2, 4), round(R10_5, 4)))
        elif corpus_name == 'ubuntu':
            print ('Test R2@1:{}, R10@1:{}, R10@2:{}, R10@5:{}'.format(round(R2_1, 4), round(R10_1, 4), 
                round(R10_2, 4), round(R10_5, 4)))
        elif corpus_name == 'e-commerce':
            curr_dev_score =  R10_1 + R10_2 + R10_5
            print ('----------------------------------------------------------------')
            print ('At epoch %d, batch %d, R10_1 is %5f, R10_2 is %5f, R10_5 is %5f' % 
                (epoch, batches_processed, R10_1, R10_2, R10_5))
        else:
            raise Exception('Wrong Corpus Name!!!')
    


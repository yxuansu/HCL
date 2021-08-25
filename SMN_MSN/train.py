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

def get_model_name(root_dir, prefix):
    for filename in os.listdir(root_dir):
        if filename.startswith(prefix):
            model_name = filename
            break
    return model_name

def hinge_loss(scores, margin):
    # y_pred: bsz x candi_num
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return torch.mean(loss)

def learn(args, total_steps, data, model, train_mode, device):
    assert train_mode in ['pretrain', 'finetune']

    directory = args.ckpt_path + '/' + train_mode + '/'
    import os
    if os.path.exists(directory):
        pass
    else: # recursively construct directory
        os.makedirs(directory, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_update_steps = int(total_steps / args.gradient_accumulation_steps) + 1
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_update_steps)
    optimizer.zero_grad()

    #--- training part ---#
    batch_size = args.batch_size
    training_data_num, dev_data_num = data.train_num, data.dev_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = int(dev_data_num / batch_size) + 1
    max_dev_score = 0.0

    batches_processed = 0
    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        loss_accumulated = 0.
        batches_processed += 1
        
        train_batch_context_id_list, train_batch_true_response_id_list, \
        train_batch_negative_response_id_list = data.get_next_train_batch(batch_size)
        train_batch_context_inp = torch.LongTensor(train_batch_context_id_list).cuda(device)
        train_batch_true_response_inp = torch.LongTensor(train_batch_true_response_id_list).cuda(device)
        train_batch_negative_response_inp = torch.LongTensor(train_batch_negative_response_id_list).cuda(device)
        train_batch_response_inp = torch.cat([train_batch_true_response_inp, train_batch_negative_response_inp], dim = 1)
        train_batch_score = model.batch_forward(train_batch_context_inp, train_batch_response_inp)
        train_loss = hinge_loss(train_batch_score, args.loss_margin)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #optimizer.step()
        #scheduler.step()
        loss_accumulated += train_loss.item()

        if (one_step+1) % args.gradient_accumulation_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if batches_processed % args.print_every == 0:
            print ('At epoch %d, batch %d, loss %.5f, max combine score is %5f' % 
                (epoch, batches_processed, loss_accumulated / batches_processed, max_dev_score))
            loss_accumulated = 0.

        if batches_processed % args.eval_every == 0:
            model.eval()
            dev_score_list, dev_label_list = [], []
            with torch.no_grad():
                for dev_step in range(dev_step_num):
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
                print ('At epoch %d, batch %d, MAP is %5f, MRR is %5f, P_1 is %5f, R10_1 is %5f, \
                        R10_2 is %5f, R10_5 is %5f' % (epoch, batches_processed, MAP, MRR, P_1, R10_1,
                        R10_2, R10_5))
                curr_dev_score = MAP + MRR + P_1 + R10_1 + R10_2 + R10_5
                print ('At epoch %d, batch %d, curr combine score is %5f' % (epoch, batches_processed, curr_dev_score))
                print ('----------------------------------------------------------------')
                    
                if curr_dev_score > max_dev_score:
                    torch.save({'args':args, 'model':model.state_dict()}, 
                        directory + '/epoch_%d_batch_%d_MAP_%.3f_MRR_%.3f_P_1_%.3f_R10_1_%.3f_R10_2_%.3f_R10_5_%.3f_combine_score_%.3f' \
                        % (epoch, batches_processed, MAP, MRR, P_1, R10_1, R10_2, R10_5, round(curr_dev_score, 3)))
                    max_dev_score = curr_dev_score
                else:
                    pass
                fileData = {}
                for fname in os.listdir(directory):
                    fileData[fname] = os.stat(directory + '/' + fname).st_mtime
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))
                if len(sortedFiles) < 1:
                    pass
                else:
                    delete = len(sortedFiles) - 1
                    for x in range(0, delete):
                        os.remove(directory + '/' + sortedFiles[x][0])
            model.train()

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
    parser.add_argument('--cutoff_threshold', type=float, default=0.8)
    parser.add_argument('--negative_selection_k', type=int, default=50)
    # model configuration
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--embedding_path', type=str)
    # learning configuration
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--lr',type=float, default=2e-4)
    parser.add_argument('--loss_margin', type=float)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--pretrain_total_steps', type=int)
    parser.add_argument('--finetune_total_steps', type=int)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)
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

    print ('---------------------------------------')
    print ('Start model pretraining...')
    negative_mode, mips_config = 'random_search', {}
    print ('Loading Data...')
    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path, args.negative_num, 
        args.dev_path, args.word2id_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode,
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    print ('Loading model...')
    model = load_matching_model(args)
    model = model.cuda(device)
    print ('Model Loaded.')

    train_mode = 'pretrain'
    learn(args, args.pretrain_total_steps, data, model, train_mode, device)
    print ('Pretraining finished.')
    print ('---------------------------------------')


    print ('Start model finetuning...')
    print ('Load the best checkpoints from pretraining...')
    pretrain_ckpt_path = args.ckpt_path + '/pretrain/'
    ckpt_name = get_model_name(pretrain_ckpt_path, prefix = 'epoch')
    model_ckpt = torch.load(pretrain_ckpt_path + '/' + ckpt_name)
    model_parameters = model_ckpt['model']

    model = load_matching_model(args)
    model.load_state_dict(model_parameters)
    model = model.cuda(device)

    print ('Loading Data...')
    negative_mode, mips_config = 'mips_search', {}
    mips_config['cutoff_threshold'] = args.cutoff_threshold
    mips_config['negative_selection_k'] = args.negative_selection_k

    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path, args.negative_num, 
        args.dev_path, args.word2id_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode,
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    train_mode = 'finetune'
    learn(args, args.finetune_total_steps, data, model, train_mode, device)
    print ('Pretraining finished.')
    print ('---------------------------------------')


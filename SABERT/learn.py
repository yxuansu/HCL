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

def hinge_loss(scores, margin):
    # y_pred: bsz x candi_num
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return torch.mean(loss)

def learn(args, total_steps, data, model, train_mode, device, multi_gpu_training):
    assert train_mode in ['pretrain', 'finetune']
    assert args.corpus_name in ['douban', 'ubuntu', 'e-commerce']

    directory = args.ckpt_path + '/' + train_mode + '/'
    import os
    if os.path.exists(directory):
        pass
    else: # recursively construct directory
        os.makedirs(directory, exist_ok=True)

    #--- training part ---#
    batch_size = args.batch_size_per_gpu * args.number_of_gpu
    training_data_num, dev_data_num = data.train_num, data.dev_num
    train_step_num = int(training_data_num / batch_size) + 1
    dev_step_num = int(dev_data_num / batch_size) + 1
    max_dev_score = 0.0

    optimizer = AdamW(model.parameters(), lr=args.lr)
    overall_update_steps = total_steps // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=overall_update_steps)
    optimizer.zero_grad()

    batches_processed = 0
    model.train()
    for one_step in range(total_steps):
        epoch = one_step // train_step_num
        loss_accumulated = 0.
        batches_processed += 1
        
        train_list_of_batch_token_id_inp, train_list_of_batch_speaker_seg_id_inp, train_list_of_batch_uttr_seg_id_inp = \
        data.get_train_next_batch(batch_size, args.negative_num)
        train_list_of_batch_token_id_inp = transform_input(train_list_of_batch_token_id_inp, device=device, use_cuda=True)
        train_list_of_batch_speaker_seg_id_inp = transform_input(train_list_of_batch_speaker_seg_id_inp, device=device, use_cuda=True)
        train_list_of_batch_uttr_seg_id_inp = transform_input(train_list_of_batch_uttr_seg_id_inp, device=device, use_cuda=True)

        '''
        if multi_gpu_training:
            train_batch_score = model.module.batch_forward(train_list_of_batch_token_id_inp, train_list_of_batch_speaker_seg_id_inp, 
                                                train_list_of_batch_uttr_seg_id_inp, is_training = True)
        else:
            train_batch_score = model.batch_forward(train_list_of_batch_token_id_inp, train_list_of_batch_speaker_seg_id_inp, 
                                                train_list_of_batch_uttr_seg_id_inp, is_training = True)
        #print (train_batch_score.size())

        train_loss = hinge_loss(train_batch_score, args.loss_margin)
        '''
        if multi_gpu_training:
            train_loss = model.module.compute_batch_loss(args.loss_margin, train_list_of_batch_token_id_inp, 
                train_list_of_batch_speaker_seg_id_inp, train_list_of_batch_uttr_seg_id_inp, is_training = True)
        else:
            train_loss = model.compute_batch_loss(args.loss_margin, train_list_of_batch_token_id_inp, 
                train_list_of_batch_speaker_seg_id_inp, train_list_of_batch_uttr_seg_id_inp, is_training = True)

        train_loss = train_loss.mean()
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
                    dev_list_of_batch_token_id_inp, dev_list_of_batch_speaker_seg_id_inp, dev_list_of_batch_uttr_seg_id_inp, \
                    dev_batch_candidate_response_label_list = data.get_next_dev_batch(batch_size)
                    dev_list_of_batch_token_id_inp = transform_input(dev_list_of_batch_token_id_inp, device=device, use_cuda=True)
                    dev_list_of_batch_speaker_seg_id_inp = transform_input(dev_list_of_batch_speaker_seg_id_inp, device=device, use_cuda=True)
                    dev_list_of_batch_uttr_seg_id_inp = transform_input(dev_list_of_batch_uttr_seg_id_inp, device=device, use_cuda=True)
                    if multi_gpu_training:
                        dev_batch_score = model.module.batch_forward(dev_list_of_batch_token_id_inp, dev_list_of_batch_speaker_seg_id_inp, 
                                                          dev_list_of_batch_uttr_seg_id_inp, is_training = False)
                    else:
                        dev_batch_score = model.batch_forward(dev_list_of_batch_token_id_inp, dev_list_of_batch_speaker_seg_id_inp, 
                                                          dev_list_of_batch_uttr_seg_id_inp, is_training = False)
                    candi_num = len(dev_list_of_batch_token_id_inp)
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
                R2_1 = compute_R2_1(valid_dev_score_list, valid_dev_label_list, n = candi_num)
                R10_1 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 1)
                R10_2 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 2)
                R10_5 = compute_R10_k(valid_dev_score_list, valid_dev_label_list, n = candi_num, k = 5)
                if args.corpus_name == 'douban':
                    curr_dev_score = MAP + MRR + P_1 + R10_1 + R10_2 + R10_5
                    print ('----------------------------------------------------------------')
                    print ('At epoch %d, batch %d, MAP is %5f, MRR is %5f, P_1 is %5f, R10_1 is %5f, \
                            R10_2 is %5f, R10_5 is %5f' % (epoch, batches_processed, MAP, MRR, P_1, R10_1,
                            R10_2, R10_5))
                    save_name = '/epoch_%d_batch_%d_MAP_%.3f_MRR_%.3f_P_1_%.3f_R10_1_%.3f_R10_2_%.3f_R10_5_%.3f_combine_score_%.3f' \
                        % (epoch, batches_processed, MAP, MRR, P_1, R10_1, R10_2, R10_5, round(curr_dev_score, 3))
                elif corpus_name == 'ubuntu':
                    curr_dev_score =  R2_1 + R10_1 + R10_2 + R10_5
                    print ('----------------------------------------------------------------')
                    print ('At epoch %d, batch %d, R2_1 is %5f, R10_1 is %5f, R10_2 is %5f, R10_5 is %5f' % 
                        (epoch, batches_processed, R2_1, R10_1, R10_2, R10_5))
                    save_name = '/epoch_%d_batch_%d_R2_1_%.3f_R10_1_%.3f_R10_2_%.3f_R10_5_%.3f_combine_score_%.3f' \
                        % (epoch, batches_processed, R2_1, R10_1, R10_2, R10_5, round(curr_dev_score, 3))
                elif corpus_name == 'e-commerce':
                    curr_dev_score =  R10_1 + R10_2 + R10_5
                    print ('----------------------------------------------------------------')
                    print ('At epoch %d, batch %d, R10_1 is %5f, R10_2 is %5f, R10_5 is %5f' % 
                        (epoch, batches_processed, R10_1, R10_2, R10_5))
                    save_name = '/epoch_%d_batch_%d_R10_1_%.3f_R10_2_%.3f_R10_5_%.3f_combine_score_%.3f' \
                        % (epoch, batches_processed, R10_1, R10_2, R10_5, round(curr_dev_score, 3))
                else:
                    raise Exception('Wrong Corpus Name!!!')

                print ('At epoch %d, batch %d, curr combine score is %5f' % (epoch, batches_processed, curr_dev_score))
                print ('----------------------------------------------------------------')     

                if curr_dev_score > max_dev_score:
                    if multi_gpu_training:
                        torch.save({'args':args, 'model':model.module.state_dict()}, directory + save_name)
                    else:
                        torch.save({'args':args, 'model':model.state_dict()}, directory + save_name)
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
    # mips configuration
    parser.add_argument('--cutoff_threshold', type=float, default=0.8)
    parser.add_argument('--negative_selection_k', type=int, default=50)
    # sampling configuration
    parser.add_argument('--negative_num', type=int, default=5, help="Number of negative samples during training.")
    # learning configuration
    #parser.add_argument('--batch_size',type=int, default=4)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=2, help="Number of available GPUs.")  
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
        help="Number of steps for gradient accumulation during training.")
    parser.add_argument('--lr',type=float, default=5e-6)
    parser.add_argument('--loss_margin', type=float, default=0.3)
    parser.add_argument('--pretrain_total_steps', type=int, default=300000)
    parser.add_argument('--finetune_total_steps', type=int, default=100000)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--ckpt_path', type=str, help="Path to save the model checkpoints.")
    return parser.parse_args()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass

    args = parse_config()
    device = torch.device('cuda')
    print (device)

    print ('---------------------------------------')
    print ('Start model pretraining...')
    negative_mode, mips_config = 'random_search', {}
    print ('Loading Data...')
    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path,
        args.dev_path, args.bert_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode, 
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    print ('Loading Model...')
    bert_model = BertModel.from_pretrained(args.bert_path)
    padding_idx = data.text_processor.padding_idx
    model = Model(bert_model, padding_idx)
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    #model = model.cuda(device)
    print ('Model Loaded.')

    train_mode = 'pretrain'
    learn(args, args.pretrain_total_steps, data, model, train_mode, device, multi_gpu_training)
    print ('Pretraining finished.')
    print ('---------------------------------------')

    print ('Start model finetuning...')
    negative_mode, mips_config = 'mips_search', {}
    mips_config['cutoff_threshold'] = args.cutoff_threshold
    mips_config['negative_selection_k'] = args.negative_selection_k

    print ('Load the best checkpoints from pretraining...')
    pretrain_ckpt_path = args.ckpt_path + '/pretrain/'
    ckpt_name = get_model_name(pretrain_ckpt_path, prefix = 'epoch')
    model_ckpt = torch.load(pretrain_ckpt_path + '/' + ckpt_name)
    model_parameters = model_ckpt['model']

    model = Model(bert_model, padding_idx)
    model.load_state_dict(model_parameters)
    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)    
    print ('Pretrained checkpoints loaded.')

    print ('Loading Data...')
    data = Data(args.train_context_path, args.train_true_response_path, args.response_index_path,
        args.dev_path, args.bert_path, args.max_uttr_num, args.max_uttr_len, mips_config, negative_mode, 
        args.train_context_vec_file, args.all_response_vec_file)
    print ('Data Loaded.')

    train_mode = 'finetune'
    learn(args, args.finetune_total_steps, data, model, train_mode, device, multi_gpu_training)
    print ('Finetuning finished.')
    print ('---------------------------------------')

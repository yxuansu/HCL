import sys
import torch
import random
import numpy as np
import progressbar
from dataclass_utlis import load_response_id_dict, load_context_data, load_dev_data, process_text
UNK, PAD = '[UNK]', '[PAD]'
# pad_context(batch_context_id_list, padding_idx)
import pickle

def load_pickle_file(in_f):
    with open(in_f, 'rb') as f:
        data =  pickle.load(f)
    return data

class Tokenizer:
    def __init__(self, word2id_path):
        self.word2id_dict, self.id2word_dict = {}, {}
        with open(word2id_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                content_list = l.strip('\n').split()
                token = content_list[0]
                idx = int(content_list[1])
                self.word2id_dict[token] = idx
                self.id2word_dict[idx] = token
        self.vocab_size = len(self.word2id_dict)
        print ('vocab size is %d' % self.vocab_size)
        self.unk_idx = self.word2id_dict[UNK]
        self.padding_idx = self.word2id_dict[PAD]

    def convert_tokens_to_ids(self, token_list):
        res_list = []
        for token in token_list:
            try:
                res_list.append(self.word2id_dict[token])
            except KeyError:
                res_list.append(self.unk_idx)
        return res_list

    def convert_ids_to_tokens(self, idx_list):
        res_list = []
        for idx in idx_list:
            try:
                res_list.append(self.id2word_dict[idx])
            except KeyError:
                res_list.append(UNK)
        return res_list

    def tokenize(self, text):
        return text.strip('\n').strip().split()

class Data:
    def __init__(self, train_context_path, train_true_response_path, response_index_path, negative_num, 
        dev_path, word2id_path, max_uttr_num, max_uttr_len, mips_config, negative_mode, train_context_vec_file, 
        all_response_vec_file):
        self.max_uttr_num, self.max_uttr_len = max_uttr_num, max_uttr_len
        self.negative_num = negative_num
        self.tokenizer = Tokenizer(word2id_path)

        self.train_context_id_list = load_context_data(train_context_path, self.max_uttr_num, 
            self.max_uttr_len, self.tokenizer)
        self.train_context_text_list = []
        with open(train_context_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                self.train_context_text_list.append(l.strip('\n'))
        self.train_num = len(self.train_context_id_list)

        self.id_response_text_dict = {}
        self.id_response_id_dict = {}
        self.index_response_text_list = []
        print ('Loading Response Index...')
        with open(response_index_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            p = progressbar.ProgressBar(len(lines))
            p.start()
            for idx in range(len(lines)):
                one_response_text = lines[idx].strip('\n')
                self.index_response_text_list.append(one_response_text)
                one_response_token_list = process_text(one_response_text, max_uttr_len, self.tokenizer)
                one_response_id_list = self.tokenizer.convert_tokens_to_ids(one_response_token_list)
                self.id_response_text_dict[idx] = one_response_text
                self.id_response_id_dict[idx] = one_response_id_list
                p.update(idx + 1)
            p.finish()
        print ('Response Index Loaded.')
        self.index_response_idx_list = [num for num in range(len(self.id_response_text_dict))]

        print ('Loading Reference Responses...')
        self.train_true_response_id_list = []
        self.train_reference_response_index_list = []
        with open(train_true_response_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_id = int(l.strip('\n'))
                self.train_reference_response_index_list.append(one_id)
                one_response_id_list = [self.id_response_id_dict[one_id]]
                self.train_true_response_id_list.append(one_response_id_list)
        assert np.array(self.train_true_response_id_list).shape == (self.train_num, 1, self.max_uttr_len)
        print ('Reference Responses Loaded.')

        self.negative_mode = negative_mode
        if negative_mode == 'random_search':
            pass
        elif negative_mode == 'mips_search':
            print ('Use Instance-Level Curriculum Learning.')
            self.cutoff_threshold = mips_config['cutoff_threshold']
            self.negative_selection_k = mips_config['negative_selection_k']
            print ('Loading context vectors...')
            self.train_context_vec = load_pickle_file(train_context_vec_file)
            assert len(self.train_context_vec) == self.train_num
            print ('Loading response index vectors...')
            self.index_response_vec = load_pickle_file(all_response_vec_file)
            assert len(self.index_response_vec) == len(self.index_response_text_list)
        else:
            raise Exception('Wrong Negative Mode!!!')

        self.dev_context_id_list, self.dev_candi_response_id_list, self.dev_candi_response_label_list = \
        load_dev_data(dev_path, self.max_uttr_num, self.max_uttr_len, self.tokenizer)

        self.dev_num = len(self.dev_context_id_list)
        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))

        self.train_idx_list = [i for i in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0
    
    def select_response(self, context_vec, true_response_vec, candidate_response_vec, candidate_response_index, 
        cutoff_threshold, top_k):
        '''
            context_vec: bsz x embed_dim
            true_response_vec: bsz x embed_dim
            candidate_response_vec: bsz x candi_num x embed_dim
            candidate_response_index: bsz x candi_num
        '''
        context_vec = torch.FloatTensor(context_vec)
        true_response_vec = torch.FloatTensor(true_response_vec)
        candidate_response_vec = torch.FloatTensor(candidate_response_vec)
        bsz, embed_dim = context_vec.size()
        _, candi_num, _ = candidate_response_vec.size()
        true_scores = torch.sum(context_vec * true_response_vec, dim = -1).view(bsz, 1)
        x = torch.unsqueeze(context_vec, 1)
        assert x.size() == torch.Size([bsz, 1, embed_dim])
        y = candidate_response_vec.transpose(1,2)
        candi_scores = torch.matmul(x, y).squeeze(1)
        assert candi_scores.size() == torch.Size([bsz, candi_num])
        score_threshold = cutoff_threshold * true_scores # do not allow too high score
        candi_scores = candi_scores - score_threshold
        candi_scores = candi_scores.masked_fill(candi_scores>0, float('-inf')) # mask out the high score part
        batch_top_scores, batch_top_indexs = torch.topk(candi_scores, k=top_k, dim = 1)
        batch_top_indexs = batch_top_indexs.cpu().tolist()
        result_index = []
        for k in range(bsz):
            one_select_index = batch_top_indexs[k]
            one_candi_index = candidate_response_index[k]
            one_res = []
            for one_id in one_select_index:
                one_res.append(one_candi_index[one_id])
            result_index.append(one_res)
        return result_index    

    def select_top_k_response(self, context_index_list, true_response_index_list, candi_response_index_list, top_k):
        '''
            context_index_list: bsz
            true_response_index_list: bsz
            candi_response_index_list: bsz x candi_num
        '''  
        batch_context_vec = [self.train_context_vec[one_id] for one_id in context_index_list]
        batch_true_response_vec = [self.index_response_vec[one_id] for one_id in true_response_index_list]
        batch_candi_response_vec = []
        for index_list in candi_response_index_list:
            one_candi_response_vec = [self.index_response_vec[one_id] for one_id in index_list]
            batch_candi_response_vec.append(one_candi_response_vec)
        batch_select_response_index = self.select_response(batch_context_vec, batch_true_response_vec, batch_candi_response_vec, 
        candi_response_index_list, self.cutoff_threshold, top_k)
        assert np.array(batch_select_response_index).shape == (len(context_index_list), top_k)
        return batch_select_response_index

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_context_id_list, batch_true_response_id_list, batch_negative_response_id_list = [], [], []
        batch_sample_response_index_list, batch_true_response_index_list = [], []
        for idx in batch_idx_list:
            # context
            one_context_id_list = self.train_context_id_list[idx]
            batch_context_id_list.append(one_context_id_list)
            # true response
            one_true_response_id_list = self.train_true_response_id_list[idx]
            batch_true_response_id_list.append(one_true_response_id_list)

            one_true_response_index = self.train_reference_response_index_list[idx]
            batch_true_response_index_list.append(one_true_response_index)

            if self.negative_mode == 'random_search':
                # random sampling negative cases
                one_negative_sample_idx_list = random.sample(self.index_response_idx_list, self.negative_num)
            elif self.negative_mode == 'mips_search':
                one_negative_sample_idx_list = random.sample(self.index_response_idx_list, self.negative_selection_k)
            else:
                raise Exception('Wrong Search Method!!!')

            while one_true_response_index in set(one_negative_sample_idx_list):
                if self.negative_mode == 'random_search':
                    # random sampling negative cases
                    one_negative_sample_idx_list = random.sample(self.index_response_idx_list, self.negative_num)
                elif self.negative_mode == 'mips_search':
                    one_negative_sample_idx_list = random.sample(self.index_response_idx_list, self.negative_selection_k)
                else:
                    raise Exception('Wrong Search Method!!!')
            batch_sample_response_index_list.append(one_negative_sample_idx_list)

        if self.negative_mode == 'random_search':
            pass
        elif self.negative_mode == 'mips_search':
            batch_context_index_list = batch_idx_list
            batch_sample_response_index_list = self.select_top_k_response(batch_context_index_list, batch_true_response_index_list, 
                                                                          batch_sample_response_index_list, self.negative_num)
        else:
            raise Exception('Wrong Search Method!!!')

        for one_sample_index_list in batch_sample_response_index_list:
            one_sample_response_id = [self.id_response_id_dict[one_neg_id] for one_neg_id in one_sample_index_list]
            batch_negative_response_id_list.append(one_sample_response_id)
        assert np.array(batch_context_id_list).shape == (batch_size, self.max_uttr_num, self.max_uttr_len)
        assert np.array(batch_true_response_id_list).shape == (batch_size, 1, self.max_uttr_len)
        assert np.array(batch_negative_response_id_list).shape == (batch_size, self.negative_num, self.max_uttr_len)
        return batch_context_id_list, batch_true_response_id_list, batch_negative_response_id_list

    def get_next_dev_batch(self, batch_size):
        '''
            batch_context_list: bsz x uttr_num x uttr_len
            batch_candidate_response_list: bsz x candi_num x uttr_len
            batch_candidate_response_label_list: bsz x candi_num
        '''
        batch_context_list, batch_candidate_response_list, batch_candidate_response_label_list = [], [], []
        if self.dev_current_idx + batch_size < self.dev_num - 1:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                one_context_id_list = self.dev_context_id_list[curr_idx] 
                batch_context_list.append(one_context_id_list)

                one_candidate_response_id_list = self.dev_candi_response_id_list[curr_idx]
                batch_candidate_response_list.append(one_candidate_response_id_list)

                one_candidate_response_label_list = self.dev_candi_response_label_list[curr_idx]
                batch_candidate_response_label_list.append(one_candidate_response_label_list)
            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                if curr_idx > self.dev_num - 1: # 对dev_current_idx重新赋值
                    curr_idx = 0
                    self.dev_current_idx = 0
                else:
                    pass
                one_context_id_list = self.dev_context_id_list[curr_idx] 
                batch_context_list.append(one_context_id_list)

                one_candidate_response_id_list = self.dev_candi_response_id_list[curr_idx]
                batch_candidate_response_list.append(one_candidate_response_id_list)

                one_candidate_response_label_list = self.dev_candi_response_label_list[curr_idx]
                batch_candidate_response_label_list.append(one_candidate_response_label_list)
            self.dev_current_idx = 0
        return batch_context_list, batch_candidate_response_list, batch_candidate_response_label_list

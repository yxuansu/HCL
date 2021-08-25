import torch
import pickle
import random
import numpy as np
import torch.nn.functional as F
from text_processor import Text_Processor
import progressbar

def load_pickle_file(in_f):
    with open(in_f, 'rb') as f:
        data =  pickle.load(f)
    return data

def load_all_response_id_list(id_response_dict, text_processor):
    print ('Loading Response Index...')
    all_response_id_list = []
    all_response_seg_list = []
    data_number = len(id_response_dict)
    p = progressbar.ProgressBar(data_number)
    p.start()
    for idx in range(data_number):
        one_response = id_response_dict[idx]
        one_response_id_list, one_response_seg_id_list = text_processor.process_response_id(one_response)
        all_response_id_list.append(one_response_id_list)
        all_response_seg_list.append(one_response_seg_id_list)
        p.update(idx)
    p.finish()
    return all_response_id_list, all_response_seg_list

class Data:
    def __init__(self, train_context_path, train_true_response_id_path, train_response_index_path,
        dev_path, word2id_path, max_uttr_num, max_uttr_len, mips_config, negative_mode, 
        train_context_vec_file, all_response_vec_file):
        '''
            max_uttr_num: maximum number of utterances contained in the context
            max_uttr_len: maximum token number of every utterance and response
        '''
        self.max_uttr_num = max_uttr_num
        self.max_uttr_len = max_uttr_len
        self.text_processor = Text_Processor(word2id_path, max_uttr_num, max_uttr_len)

        self.id_response_text_dict = {}
        self.index_response_text_list = []
        with open(train_response_index_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            data_num = len(lines)
            for idx in range(data_num):
                one_response = lines[idx].strip('\n')
                self.id_response_text_dict[idx] = one_response
                self.index_response_text_list.append(one_response)
        self.index_size = len(self.id_response_text_dict)
        print ('Size of the response index is %d' % self.index_size)
        self.all_response_id_list, self.all_response_seg_list = load_all_response_id_list(self.id_response_text_dict, self.text_processor)
        assert len(self.all_response_id_list) == self.index_size
        self.index_idx_list = [num for num in range(self.index_size)]
        self.index_response_idx_list = [num for num in range(len(self.id_response_text_dict))]

        self.train_context_id_list, self.train_context_seg_list = [], []
        print ('Start loading training context...')
        with open(train_context_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            p = progressbar.ProgressBar(len(lines))
            p.start()
            data_idx = 0
            for l in lines:
                context_text = l.strip('\n')
                one_context_id_list, one_context_seg_list = self.text_processor.process_context_id(context_text)
                self.train_context_id_list.append(one_context_id_list)
                self.train_context_seg_list.append(one_context_seg_list)
                p.update(data_idx+1)
                data_idx += 1
            p.finish()

        self.train_true_response_id_list, self.train_true_response_seg_id_list = [], []
        self.train_true_response_index_list = []
        print ('Start loading training true response...')
        with open(train_true_response_id_path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            p = progressbar.ProgressBar(len(lines))
            p.start()
            data_idx = 0
            for l in lines:
                one_response_idx = int(l.strip('\n'))
                self.train_true_response_index_list.append(one_response_idx)
                one_response_id_list = self.all_response_id_list[one_response_idx]
                one_response_seg_id_list = self.all_response_seg_list[one_response_idx]
                self.train_true_response_id_list.append(one_response_id_list)
                self.train_true_response_seg_id_list.append(one_response_seg_id_list)
                p.update(data_idx+1)
                data_idx += 1
            p.finish()
        assert len(self.train_context_id_list) == len(self.train_true_response_id_list)
        self.train_num = len(self.train_context_id_list)

        self.negative_mode = negative_mode
        if negative_mode == 'random_search':
            print ('Do not use Instance-Level Curriculum Learning. Just Random Search.')
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
            raise Exception('Wrong negative mode!!!')

        self.dev_context_id_list, self.dev_context_seg_id_list, self.dev_candi_response_id_list, \
        self.dev_candi_response_seg_id_list, self.dev_candi_response_label_list = self.load_dev_data(dev_path)
        self.dev_num = len(self.dev_context_id_list)

        print ('train number is %d, dev number is %d' % (self.train_num, self.dev_num))
        self.train_idx_list = [j for j in range(self.train_num)]
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.dev_current_idx = 0

    def select_response(self, context_vec, true_response_vec, candidate_response_vec, 
        candidate_response_index, cutoff_threshold, top_k):
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

    def get_train_next_batch(self, batch_size, neg_num):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_context_id_list, batch_context_seg_id_list = [], []
        batch_true_response_id_list, batch_true_response_seg_id_list = [], []
        batch_true_response_index_list = []
        batch_sample_response_index_list = []
        for idx in batch_idx_list:
            one_context_id_list, one_context_seg_id_list = self.train_context_id_list[idx], self.train_context_seg_list[idx]
            batch_context_id_list.append(one_context_id_list)
            batch_context_seg_id_list.append(one_context_seg_id_list)

            one_true_response_id_list, one_true_response_seg_id_list = \
            self.train_true_response_id_list[idx], self.train_true_response_seg_id_list[idx]
            batch_true_response_id_list.append(one_true_response_id_list)
            batch_true_response_seg_id_list.append(one_true_response_seg_id_list)

            one_true_response_index = self.train_true_response_index_list[idx]
            batch_true_response_index_list.append(one_true_response_index)

            if self.negative_mode == 'random_search':
                random_idx_list = random.sample(self.index_idx_list, neg_num)
            elif self.negative_mode == 'mips_search':
                random_idx_list = random.sample(self.index_response_idx_list, self.negative_selection_k)
            else:
                raise Exception('Wrong Search Method!!!')

            while one_true_response_index in random_idx_list: 
                if self.negative_mode == 'random_search':
                    # random sampling negative cases
                    random_idx_list = random.sample(self.index_idx_list, neg_num)
                elif self.negative_mode == 'mips_search':
                    random_idx_list = random.sample(self.index_response_idx_list, self.negative_selection_k)
                else:
                    raise Exception('Wrong Search Method!!!')
            batch_sample_response_index_list.append(random_idx_list)

        list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, list_of_batch_uttr_seg_id_inp = [], [], []
        true_batch_token_id_list, true_batch_speaker_seg_id_list, true_batch_uttr_seg_id_list = \
        self.text_processor.process_batch_result(batch_context_id_list, batch_context_seg_id_list, 
                                                batch_true_response_id_list, batch_true_response_seg_id_list)
        list_of_batch_token_id_inp.append(true_batch_token_id_list)
        list_of_batch_speaker_seg_id_inp.append(true_batch_speaker_seg_id_list)
        list_of_batch_uttr_seg_id_inp.append(true_batch_uttr_seg_id_list)

        # instance level curriculum learning
        if self.negative_mode == 'random_search':
            pass
        elif self.negative_mode == 'mips_search':
            batch_context_index_list = batch_idx_list
            batch_sample_response_index_list = self.select_top_k_response(batch_context_index_list, 
                batch_true_response_index_list, batch_sample_response_index_list, neg_num)
        else:
            raise Exception('Wrong Search Method!!!')

        # batch_sample_response_index_list: bsz x neg_num
        assert len(batch_sample_response_index_list) == batch_size
        assert len(batch_sample_response_index_list[0]) == neg_num
        for k in range(neg_num):
            one_batch_neg_response_id_list, one_batch_neg_response_seg_id_list = [], []
            for j in range(batch_size):
                one_neg_index = batch_sample_response_index_list[j][k]
                one_neg_response_id = self.all_response_id_list[one_neg_index]
                one_batch_neg_response_id_list.append(one_neg_response_id)
                one_neg_response_seg_id = self.all_response_seg_list[one_neg_index]
                one_batch_neg_response_seg_id_list.append(one_neg_response_seg_id)
            neg_batch_token_id_list, neg_batch_speaker_seg_id_list, neg_batch_uttr_seg_id_list = \
            self.text_processor.process_batch_result(batch_context_id_list, batch_context_seg_id_list, 
                                                    one_batch_neg_response_id_list, one_batch_neg_response_seg_id_list)
            list_of_batch_token_id_inp.append(neg_batch_token_id_list)
            list_of_batch_speaker_seg_id_inp.append(neg_batch_speaker_seg_id_list)
            list_of_batch_uttr_seg_id_inp.append(neg_batch_uttr_seg_id_list)
        return list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, list_of_batch_uttr_seg_id_inp

    def load_dev_data(self, path):
        '''
            each response candidate list contains 10 responses
            each candidate response label list contains 10 labels
        '''
        all_context_id_list, all_context_seg_id_list = [], []
        all_candi_response_id_list, all_candi_response_seg_id_list = [], []
        all_candi_response_label_list = []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            test_data_num = int(len(lines) / 10)
            print ('test data number is %d' % test_data_num)
            for i in range(test_data_num):
                if i % int(test_data_num / 10) == 0:
                    print ('%d test instances have been loaded' % i)
                batch_text_list = lines[i*10:(i+1)*10]
                batch_text_list = [text.strip('\n') for text in batch_text_list]
                one_context_text_list = batch_text_list[0].strip('\n').split('\t')[1:-1]
                one_context_text = '\t'.join(one_context_text_list).strip('\t')
                one_context_id_list, one_context_seg_id_list = self.text_processor.process_context_id(one_context_text)
                all_context_id_list.append(one_context_id_list)
                all_context_seg_id_list.append(one_context_seg_id_list)

                start_idx = i*10
                one_candi_response_id_list, one_candi_response_seg_id_list = [], []
                one_candi_response_label_list = []
                for candi_idx in range(start_idx, start_idx + 10):
                    one_line = lines[candi_idx]
                    one_line_content_list = one_line.strip('\n').split('\t')
                    one_candi_response_label = int(one_line_content_list[0])
                    one_candi_response_label_list.append(one_candi_response_label)
                    one_candi_response_text = one_line_content_list[-1]
                    one_rep_id, one_seg_id = self.text_processor.process_response_id(one_candi_response_text)
                    one_candi_response_id_list.append(one_rep_id)
                    one_candi_response_seg_id_list.append(one_seg_id)

                all_candi_response_id_list.append(one_candi_response_id_list)
                all_candi_response_seg_id_list.append(one_candi_response_seg_id_list)
                all_candi_response_label_list.append(one_candi_response_label_list)
        return all_context_id_list, all_context_seg_id_list, all_candi_response_id_list, all_candi_response_seg_id_list, all_candi_response_label_list 

    def get_next_dev_batch(self, batch_size):
        '''
            batch_context_id_list, batch_context_seg_list: bsz x context_len
            batch_candi_response_id_list, batch_candi_response_seg_list: bsz x candi_num x response_len
        '''
        batch_context_id_list, batch_context_seg_id_list, batch_candi_response_id_list, \
        batch_candi_response_seg_id_list = [], [], [], []
        batch_candi_response_label_list = []

        if self.dev_current_idx + batch_size < self.dev_num - 1:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                # context input
                one_context_id = self.dev_context_id_list[curr_idx]
                batch_context_id_list.append(one_context_id)
                one_context_seg_id = self.dev_context_seg_id_list[curr_idx]
                batch_context_seg_id_list.append(one_context_seg_id)

                # candidate response input
                one_candi_response_id = self.dev_candi_response_id_list[curr_idx]
                batch_candi_response_id_list.append(one_candi_response_id)
                one_candi_response_seg_id = self.dev_candi_response_seg_id_list[curr_idx]
                batch_candi_response_seg_id_list.append(one_candi_response_seg_id)

                one_candi_label = self.dev_candi_response_label_list[curr_idx]
                batch_candi_response_label_list.append(one_candi_label)
            self.dev_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = self.dev_current_idx + i
                if curr_idx > self.dev_num - 1: # 对dev_current_idx重新赋值
                    curr_idx = 0
                    self.dev_current_idx = 0
                else:
                    pass
                one_context_id = self.dev_context_id_list[curr_idx]
                batch_context_id_list.append(one_context_id)
                one_context_seg_id = self.dev_context_seg_id_list[curr_idx]
                batch_context_seg_id_list.append(one_context_seg_id)

                # candidate response input
                one_candi_response_id = self.dev_candi_response_id_list[curr_idx]
                batch_candi_response_id_list.append(one_candi_response_id)
                one_candi_response_seg_id = self.dev_candi_response_seg_id_list[curr_idx]
                batch_candi_response_seg_id_list.append(one_candi_response_seg_id)

                one_candi_label = self.dev_candi_response_label_list[curr_idx]
                batch_candi_response_label_list.append(one_candi_label)
            self.dev_current_idx = 0

        list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, list_of_batch_uttr_seg_id_inp = [], [], []
        candi_num = len(batch_candi_response_label_list[0])
        for k in range(candi_num):
            one_batch_response_id_list, one_batch_response_seg_id_list = [], []
            for j in range(batch_size):
                one_response_id = batch_candi_response_id_list[j][k]
                one_batch_response_id_list.append(one_response_id)
                one_response_seg_id = batch_candi_response_seg_id_list[j][k]
                one_batch_response_seg_id_list.append(one_response_seg_id)
            batch_token_id_list, batch_speaker_seg_id_list, batch_uttr_seg_id_list = \
            self.text_processor.process_batch_result(batch_context_id_list, batch_context_seg_id_list, 
                                                    one_batch_response_id_list, one_batch_response_seg_id_list)
            list_of_batch_token_id_inp.append(batch_token_id_list)
            list_of_batch_speaker_seg_id_inp.append(batch_speaker_seg_id_list)
            list_of_batch_uttr_seg_id_inp.append(batch_uttr_seg_id_list)
        return list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, list_of_batch_uttr_seg_id_inp, batch_candi_response_label_list

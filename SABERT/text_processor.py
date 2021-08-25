import random
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
PAD, SEP, CLS, EOU = '[PAD]', '[SEP]', '[CLS]', '[unused1]'

class Text_Processor:
    def __init__(self, word2id_path, max_uttr_num, max_uttr_len):
        self.tokenizer = BertTokenizer.from_pretrained(word2id_path) # word2id_path refers to the bert path
        self.padding_idx, self.sep_idx, self.cls_idx, self.eou_idx = \
        self.tokenizer.convert_tokens_to_ids([PAD, SEP, CLS, EOU])
        self.max_uttr_num, self.max_uttr_len = max_uttr_num, max_uttr_len

    def process_context_id(self, context_text):
        context_text_list = context_text.strip('\n').split('\t')
        context_token_list = [CLS]
        context_seg_list = [0]
        context_text_list = context_text_list[-self.max_uttr_num:] 
        usr_id = 0
        for text_idx in range(len(context_text_list)):
            text = context_text_list[text_idx]
            if text_idx == len(context_text_list) - 1: # the last utterance
                one_token_list = self.tokenizer.tokenize(text)[:self.max_uttr_len] + [EOU] + [SEP]
            else:
                one_token_list = self.tokenizer.tokenize(text)[:self.max_uttr_len] + [EOU]
            context_token_list.extend(one_token_list)
            one_seg_list = [usr_id for _ in range(len(one_token_list))]
            context_seg_list.extend(one_seg_list)
            usr_id = 1 - usr_id
        context_id_list = self.tokenizer.convert_tokens_to_ids(context_token_list)
        return context_id_list, context_seg_list

    def process_response_id(self, response_text):
        response_token_list = self.tokenizer.tokenize(response_text.strip('\n').strip())[:self.max_uttr_len]
        response_token_list = response_token_list + [SEP]
        response_seg_list = [2 for _ in response_token_list] # this part will be masked out before BERT processing
        return self.tokenizer.convert_tokens_to_ids(response_token_list), response_seg_list

    def pad_token_id_list(self, batch_token_id_list):
        len_list = [len(item) for item in batch_token_id_list]
        max_len = max(len_list)
        res_list = []
        for item in batch_token_id_list:
            len_diff = max_len - len(item)
            one_res = item + [self.padding_idx for _ in range(len_diff)]
            res_list.append(one_res)
        return res_list

    def pad_seg_id_list(self, batch_seg_id_list):
        len_list = [len(item) for item in batch_seg_id_list]
        max_len = max(len_list)
        res_list = []
        for item in batch_seg_id_list:
            len_diff = max_len - len(item)
            one_res = item + [2 for _ in range(len_diff)] # 2 is for masking purpose
            res_list.append(one_res)
        return res_list

    def pad_uttr_id_list(self, batch_uttr_id_list):
        len_list = [len(item) for item in batch_uttr_id_list]
        max_len = max(len_list)
        res_list = []
        for item in batch_uttr_id_list:
            len_diff = max_len - len(item)
            one_res = item + [1 for _ in range(len_diff)] # 2 is for masking purpose
            res_list.append(one_res)
        return res_list

    def process_batch_result(self, batch_context_token_id_list, batch_context_seg_id_list, 
        batch_response_token_id_list, batch_response_seg_id_list): 
        # batch_context_token_id_list: bsz x context_len
        # batch_response_token_id_list: bsz x response_len
        batch_token_id_list, batch_seg_id_list = [], []
        batch_uttr_response_seg_list = []
        bsz = len(batch_context_token_id_list)
        for k in range(bsz):
            one_context_token_id_list, one_context_seg_id_list = batch_context_token_id_list[k], \
            batch_context_seg_id_list[k]
            one_uttr_seg_list = [0 for _ in range(len(one_context_token_id_list))]

            one_response_token_id_list, one_response_seg_id_list = batch_response_token_id_list[k], \
            batch_response_seg_id_list[k]

            one_token_id_res = one_context_token_id_list + one_response_token_id_list
            one_uttr_seg_list += [1 for _ in range(len(one_response_token_id_list))]
            one_seg_id_res = one_context_seg_id_list + one_response_seg_id_list
            assert len(one_token_id_res) == len(one_seg_id_res)
            assert len(one_uttr_seg_list) == len(one_seg_id_res)
            batch_token_id_list.append(one_token_id_res)
            batch_seg_id_list.append(one_seg_id_res)
            batch_uttr_response_seg_list.append(one_uttr_seg_list)
        return self.pad_token_id_list(batch_token_id_list), self.pad_seg_id_list(batch_seg_id_list), \
        self.pad_uttr_id_list(batch_uttr_response_seg_list)

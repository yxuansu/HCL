import math
import torch
from torch import nn
import torch.nn.functional as F
from transformer import Seg_Embedding

def hinge_loss(scores, margin):
    # y_pred: bsz x candi_num
    loss = torch.nn.functional.relu(margin - (torch.unsqueeze(scores[:, 0], -1) - scores[:, 1:]))
    return loss
    #return torch.mean(loss)

class Model(nn.Module):
    def __init__(self, bert_model, padding_idx):
        super(Model, self).__init__()
        self.bert_model = bert_model
        self.hidden_size = self.bert_model.config.hidden_size
        self.speaker_seg_embedding = Seg_Embedding(3, self.hidden_size)
        self.padding_idx = padding_idx
        self.final_linear = nn.Linear(self.hidden_size, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, batch_token_id_inp, batch_speaker_seg_id_inp, batch_uttr_seg_id_inp, is_training):
        batch_speaker_seg_embeddings = self.speaker_seg_embedding(batch_speaker_seg_id_inp)
        #batch_mask = 1 - batch_token_id_inp.eq(self.padding_idx)
        batch_mask = ~batch_token_id_inp.eq(self.padding_idx)
        if is_training:
            batch_representation, _ = self.bert_model(batch_token_id_inp, batch_speaker_seg_embeddings, batch_speaker_seg_id_inp, 
                token_type_ids=batch_uttr_seg_id_inp, attention_mask = batch_mask, output_all_encoded_layers = False)
        else:
            batch_representation, _ = self.bert_model.work(batch_token_id_inp, batch_speaker_seg_embeddings, batch_speaker_seg_id_inp, 
                token_type_ids=batch_uttr_seg_id_inp, attention_mask = batch_mask, output_all_encoded_layers = False)
        bsz, seqlen, _ = batch_representation.size()
        batch_cls_vec = batch_representation.transpose(0,1)[0]
        assert batch_cls_vec.size() == torch.Size([bsz, self.hidden_size])
        logits = self.final_linear(batch_cls_vec)
        assert logits.size() == torch.Size([bsz, 1])
        return logits

    def batch_forward(self, list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, list_of_batch_uttr_seg_id_inp, is_training):
        # list_of_batch_token_id_inp: each item has size of bsz x seqlen_i
        # length of list_of_batch_token_id_inp is candi_num
        item_num = len(list_of_batch_token_id_inp)
        bsz = list_of_batch_token_id_inp[0].size()[0]
        batch_score_list = []
        for k in range(item_num):
            batch_token_id_inp = list_of_batch_token_id_inp[k]
            batch_speaker_seg_id_inp = list_of_batch_speaker_seg_id_inp[k]
            batch_uttr_seg_id_inp = list_of_batch_uttr_seg_id_inp[k]
            one_score = self.forward(batch_token_id_inp, batch_speaker_seg_id_inp, batch_uttr_seg_id_inp, is_training)
            batch_score_list.append(one_score)
        batch_scores = torch.cat(batch_score_list, dim = -1)
        assert batch_scores.size() == torch.Size([bsz, item_num])
        return batch_scores

    def compute_batch_loss(self, margin, list_of_batch_token_id_inp, list_of_batch_speaker_seg_id_inp, 
        list_of_batch_uttr_seg_id_inp, is_training):
        # list_of_batch_token_id_inp: each item has size of bsz x seqlen_i
        # length of list_of_batch_token_id_inp is candi_num
        item_num = len(list_of_batch_token_id_inp)
        bsz = list_of_batch_token_id_inp[0].size()[0]
        batch_score_list = []
        for k in range(item_num):
            batch_token_id_inp = list_of_batch_token_id_inp[k]
            batch_speaker_seg_id_inp = list_of_batch_speaker_seg_id_inp[k]
            batch_uttr_seg_id_inp = list_of_batch_uttr_seg_id_inp[k]
            one_score = self.forward(batch_token_id_inp, batch_speaker_seg_id_inp, batch_uttr_seg_id_inp, is_training)
            batch_score_list.append(one_score)
        batch_scores = torch.cat(batch_score_list, dim = -1)
        assert batch_scores.size() == torch.Size([bsz, item_num])
        batch_loss = hinge_loss(batch_scores, margin)
        return batch_loss
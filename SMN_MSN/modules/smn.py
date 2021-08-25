# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import pickle

class SMN(nn.Module):
    def __init__(self, emb_matrix, match_type=0, max_num_utterances=10):
        super(SMN, self).__init__()
        self.emb_dim = 200
        self.hidden_dim = 200
        self.vocab_size = len(emb_matrix) #400000
        self.layers_num = 2
        self.match_type = match_type
        self.max_num_utterances = max_num_utterances

        def gru_weight_init(weights):
            for l in weights:
                for weight in l:
                    if len(weight.shape) >= 2: nn.init.orthogonal_(weight)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        # utterance的gru
        self.utterance_GRU = nn.GRU(self.emb_dim, self.hidden_dim, bidirectional=False, batch_first=True)
        gru_weight_init(self.utterance_GRU.all_weights)

        # response的gru
        self.response_GRU = nn.GRU(self.emb_dim, self.hidden_dim, bidirectional=False, batch_first=True)
        gru_weight_init(self.response_GRU.all_weights)
        # 卷积层的定义
        self.conv2d = nn.Conv2d(2, 8, kernel_size=(3, 3))
        nn.init.kaiming_normal_(self.conv2d.weight)
        # 进入Matching Accumulationmo模块前，进行维度映射
        self.linear = nn.Linear(16 * 16 * 8, 50)
        nn.init.xavier_uniform_(self.linear.weight)

        self.matrix_A = nn.Parameter(torch.randn([self.hidden_dim, self.hidden_dim]))

        if self.match_type == 1:
            self.static_match_weight = nn.Parameter(torch.randn([10, 1]))
        elif self.match_type == 2:
            self.match_W11_matrix = nn.Parameter(torch.randn([50, self.hidden_dim]))
            self.match_W12_matrix = nn.Parameter(torch.randn([50, 50]))
            self.match_b1 = nn.Parameter(torch.zeros([50]))
            self.ts = nn.Parameter(torch.randn([50, 1]))


        self.final_GRU = nn.GRU(50, 50, bidirectional=False, batch_first=True)
        gru_weight_init(self.final_GRU.all_weights)

        #self.final_linear = nn.Linear(50, 1)
        self.final_linear = nn.Linear(50, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, utterance, response):
        '''
            utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
            response:(self.batch_size, self.max_sentence_len)
        '''
        # (batch_size,10,50)-->(batch_size,10,50,200)
        all_utterance_embeddings = self.embedding(utterance)
        response_embeddings = self.embedding(response)

        # pytorch:(batch_size,10,50,200)-->(10,batch_size,50,200)
        all_utterance_embeddings = all_utterance_embeddings.permute(1, 0, 2, 3)

        response_GRU_embeddings, _ = self.response_GRU(response_embeddings) # batch_size, 50, 200 --> batch_size, 50, 200
        response_embeddings = response_embeddings.permute(0, 2, 1) # batch_size, 50, 200 --> batch_size, 200, 50
        response_GRU_embeddings = response_GRU_embeddings.permute(0, 2, 1) # batch_size, 50, 200 --> batch_size, 200, 50
        matching_vectors = []
        utterance_hidden = []

        for utterance_embeddings in all_utterance_embeddings:
            matrix1 = torch.matmul(utterance_embeddings, response_embeddings) #    --> batch_size, 50, 50

            utterance_GRU_embeddings, uhidden = self.utterance_GRU(utterance_embeddings) # batch_size, 50, 200 --> batch_size, 50, 200
            utterance_hidden.append(torch.squeeze(uhidden))
            matrix2 = torch.einsum('aij,jk->aik', (utterance_GRU_embeddings, self.matrix_A))
            matrix2 = torch.matmul(matrix2, response_GRU_embeddings)

            matrix = torch.stack([matrix1, matrix2], dim=1)
            # matrix:(batch_size,channel,seq_len,embedding_size)
            conv_layer = self.conv2d(matrix)
            # add activate function
            conv_layer = F.relu(conv_layer)

            pooling_layer = F.max_pool2d(conv_layer, kernel_size=3, stride=3)
            # flatten
            pooling_layer = pooling_layer.view(pooling_layer.size(0), -1)
            matching_vector = self.linear(pooling_layer)
            # add activate function
            matching_vector = F.tanh(matching_vector)
            matching_vectors.append(matching_vector)
        # Matching Accumulation
        gru_output, last_hidden = self.final_GRU(torch.stack(matching_vectors, dim=1)) # batch_size, 10, 50 --> (batch_size, 10, 50) (1, batch_size, 50)
        if self.match_type == 0:
            matching_output = last_hidden
        elif self.match_type == 1:
            matching_output = torch.sum(gru_output * self.static_match_weight, 1)
        else:
            f1 = torch.matmul(torch.stack(utterance_hidden, dim=1), self.match_W11_matrix.permute(1, 0))
            f2 = torch.matmul(gru_output, self.match_W12_matrix.permute(1, 0))
            ti = F.tanh(f1 + f2 + self.match_b1)
            att = F.softmax(torch.matmul(ti, self.ts), 1) # batch_size, 10, 1
            matching_weight = torch.mul(att, gru_output)
            matching_output = torch.sum(matching_weight, 1)

        # 输出层
        #print (matching_output.size())
        # 输出层
        logits = self.final_linear(torch.squeeze(matching_output))

        # use CrossEntropyLoss,this loss function would accumulate softmax
        # y_pred = F.softmax(logits, -1)
        y_pred = logits # bsz x 1
        return y_pred

    def batch_forward(self, utterance, batch_response):
        '''
            utterance: bsz x max_uttr_num x max_uttr_len
            batch_response: bsz x response_candi_num x max_uttr_len
        '''
        batch_score_list = []
        bsz, candi_num, max_uttr_len = batch_response.size()
        batch_response_list = torch.unbind(batch_response, dim = 1)
        assert len(batch_response_list) == candi_num
        for idx in range(candi_num):
            one_batch_response = batch_response_list[idx]
            assert one_batch_response.size() == torch.Size([bsz, max_uttr_len])
            one_score = self.forward(utterance, one_batch_response)
            assert one_score.size() == torch.Size([bsz, 1])
            batch_score_list.append(one_score)
        batch_score = torch.cat(batch_score_list, dim = -1)
        assert batch_score.size() == torch.Size([bsz, candi_num])
        return batch_score

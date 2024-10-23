# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
from recbole.model.loss import BPRLoss


class PNN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def BPR_loss(pos_cos, neg_cos):
        return torch.log(1 + torch.exp(neg_cos.mean(dim=-1) - pos_cos).sum(dim=1))
    @staticmethod
    def Euclidean(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()
    @staticmethod
    def item_loss(x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x.mean(dim=-2).squeeze(-2), p=2).pow(2).mul(-2).exp().mean().log()

    def __init__(self, config, dataset):
        super(PNN, self).__init__(config, dataset)

        # Get user history interacted items
        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.alpha = config["alpha"]
        self.beta = config['beta']
        self.negative_weight = config["negative_weight"]
        self.gamma = config["gamma"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        self.reg_weight = config["reg_weight"]
        self.aggregator = config["aggregator"]
        if self.aggregator not in ["mean", "user_attention", "self_attention"]:
            raise ValueError(
                "aggregator must be mean, user_attention or self_attention"
            )
        self.history_len = torch.max(self.history_item_len, dim=0)

        # user embedding matrix
        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        # item embedding matrix
        self.item_emb = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        # feature space mapping matrix of user and item
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
        # dropout
        self.lambdat = nn.Linear(self.embedding_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.require_pow = config["require_pow"]
        # l2 regularization loss
        self.reg_loss = EmbLoss()
        self.BPR = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # get the mask
        self.item_emb.weight.data[0, :] = 0
    


    def get_UI_aggregation(self, user_e, history_item_e, history_len):

        # [user_num, max_history_len, embedding_size]
        key = self.W_k(history_item_e)
        attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
        e_attention = torch.exp(attention)
        mask = (history_item_e.sum(dim=-1) != 0).int()
        e_attention = e_attention * mask
        # [user_num, max_history_len]
        attention_weight = e_attention / (
            e_attention.sum(dim=1, keepdim=True) + 1.0e-10
        )
        # [user_num, embedding_size]
        out = torch.matmul(attention_weight.unsqueeze(1), history_item_e).squeeze(1)
    # Combined vector of user and item sequences
        out = self.UI_map(out)
        g = self.gamma
        UI_aggregation_e = g * user_e + (1 - g) * out
        return UI_aggregation_e

    def score(self, user_e, item_e):
        score = torch.sum(torch.mul(user_e, item_e), axis=-1)
        return score

    def forward(self, user, pos_item, history_item, history_len, neg_item_seq):

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        pos_item_e = self.item_emb(pos_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)
        # [nuser_num, neg_seq_len, embedding_size]
        neg_item_seq_e = self.item_emb(neg_item_seq)

        pos_cos = self.score(user_e, pos_item_e)
        neg_cos = self.score(user_e.unsqueeze(dim=1), neg_item_seq_e)


        #semi-supervised learning
        
        max_indices = torch.max(neg_cos, dim=1)[1].detach()  # [batch_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)
        lambdat = torch.sigmoid(self.lambdat(UI_aggregation_e).mean())

        hard_item = torch.gather(neg_item_seq, dim=1, index=max_indices.unsqueeze(-1)).squeeze()
        hard_neg_e = self.item_emb(hard_item)
        hard_neg_cos = self.score(user_e, hard_neg_e)
        bpr_loss = self.BPR(pos_cos,hard_neg_cos)
        
        # classifier

        min_indices = torch.min(neg_cos, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_item_seq, dim=1, index=min_indices.unsqueeze(-1)).squeeze()
        true_neg_e = self.item_emb(neg_item)
        true_neg_cos = self.score(user_e, true_neg_e)

        pos_dir = torch.sign(pos_item_e.unsqueeze(dim=1))
        neg_dir = torch.sign(true_neg_e.unsqueeze(dim=1))

        pos_random_noise = torch.rand(pos_dir.shape).to(self.device)
        neg_random_noise = torch.rand(neg_dir.shape).to(self.device) 

        pos_random_noise = torch.nn.functional.normalize(pos_random_noise, p=2, dim=-1) * 0.1               
        neg_random_noise = torch.nn.functional.normalize(neg_random_noise, p=2, dim=-1) * 0.1

        noise1 = pos_random_noise + neg_item_seq_e
        noise2 = neg_random_noise + neg_item_seq_e



        l_class = self.Euclidean(noise1 , noise2)
        l_item = self.item_loss(neg_item_seq_e) / 2
        amb_loss = self.alpha * l_class + self.beta * l_item

        l_rank = self.BPR(pos_cos,neg_cos.mean(dim=-1)) + self.BPR(neg_cos.mean(dim=-1),true_neg_cos)


        
        final_loss = lambdat * bpr_loss  + (1-lambdat)*(l_rank + self.alpha * l_class + self.beta * l_item )

        # l2 regularization loss
        reg_loss = self.reg_loss(
            user_e,
            pos_item_e,
            # history_item_e,
            neg_item_seq_e,
            require_pow=self.require_pow,
        )

        loss = final_loss + self.reg_weight * reg_loss.sum()
        return loss

    def calculate_loss(self, interaction):

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]

        # get your neg items based on your sampling strategy
        neg_item_seq = None
        
        user_number = int(len(user) / self.neg_seq_len)
        # user's id
        user = user[0:user_number]
        # historical transaction record
        history_item = self.history_item_id[user]
        # positive item's id
        pos_item = pos_item[0:user_number]
        # history_len
        history_len = self.history_item_len[user]

        loss = self.forward(user, pos_item, history_item, history_len, neg_item_seq)
        return loss



    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        # user_e, item_e = self.forward(user, item)
        user_e = self.user_emb(user)
    #     # [user_num, embedding_size]
        item_e = self.item_emb(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_emb(user)
        all_item_e = self.item_emb.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)



import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class ANS(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ANS, self).__init__(config, dataset)
        self.emb_size = config["embedding_size"]
        self.embedding_size = config["embedding_size"]
        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.UI_map = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_k = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.hard_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.conf_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.easy_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.margin_model = nn.Linear(self.emb_size, 1).to(self.device)
        self.neg_weight = nn.Linear(self.neg_seq_len, self.neg_seq_len).to(self.device)
        self.eps=config["eps"]
        self.gamma=config["gamma"]
    

        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    def get_UI_aggregation(self, user_e, negtaive_item_e, history_len):

            # [user_num, max_history_len, embedding_size]
        key = self.W_k(negtaive_item_e)
        attention = torch.matmul(key, user_e.unsqueeze(2)).squeeze(2)
        e_attention = torch.exp(attention)
        mask = (negtaive_item_e.sum(dim=-1) != 0).int()
        e_attention = e_attention * mask
        # [user_num, max_history_len]
        attention_weight = e_attention / (
            e_attention.sum(dim=1, keepdim=True) + 1.0e-10
        )
        # [user_num, embedding_size]
        negative_preference_attention = torch.sigmoid(self.neg_weight(attention_weight))
        out = torch.matmul(negative_preference_attention.unsqueeze(1), negtaive_item_e).squeeze(1)
    # Combined vector of user and item sequences
        out = self.UI_map(out)
        # g = self.gamma
        # UI_aggregation_e = g * user_e + (1 - g) * out
        # return UI_aggregation_e
        return out
    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        embs = lightgcn_all_embeddings
        return embs[:self.n_users, :], embs[self.n_users:, :]
    



    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        neg_item_seq = neg_item.reshape((self.neg_seq_len, -1))
        neg_item_seq = neg_item_seq.T

        neg_item = neg_item_seq
        user_number = int(len(user) / self.neg_seq_len)
        user = user[0:user_number]
        pos_item = pos_item[0:user_number]

        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]




        s_e = u_embeddings
        p_e = pos_embeddings
        n_e = neg_embeddings
        u_e = u_embeddings.mean(dim=1)
        pos_e = pos_embeddings.mean(dim=1)
        neg_e = neg_embeddings.mean(dim=1)
        batch_size = user.shape[0]
        
        gate_neg_hard = torch.sigmoid(self.item_gate(n_e) * self.user_gate(s_e).unsqueeze(1))
        n_hard =  n_e * gate_neg_hard
        n_easy =  n_e - n_hard
        
        p_hard =  p_e.unsqueeze(1) * gate_neg_hard
        p_easy =  p_e.unsqueeze(1) - p_hard
    
        import torch.nn.functional as F
        distance = torch.mean(F.pairwise_distance(n_hard, p_hard, p=2).squeeze(dim=1))
        temp = torch.norm(torch.mul(p_easy, n_easy),dim=-1)
        orth = torch.mean(torch.sum(temp,axis=-1))

        margin = torch.sigmoid(1/self.margin_model(n_hard * p_hard))

        random_noise = torch.rand(n_easy.shape).to(self.device)
        magnitude = torch.nn.functional.normalize(random_noise, p=2, dim=-1) * margin *0.1
        direction = torch.sign(p_easy - n_easy)
        noise = torch.mul(direction,magnitude)
        n_easy_syth = noise + n_easy


        u_neg_pre = self.get_UI_aggregation(u_e.squeeze(-2),n_e.squeeze(-2),self.neg_seq_len).unsqueeze(dim=1)
        

        # u_neg_pre 
        # # print(s_e.shape)
        # # print(n_hard.shape)
        # print(u_neg_pre.shape)
        # exit(0)
        # print(n_hard.shape)
        # print(u_neg_pre.shape)
        n_hard_syth = 0.1 * n_hard * torch.sigmoid(n_hard * u_neg_pre.unsqueeze(dim=1)) + 0.9 * n_hard
        
        a = torch.norm(torch.mul(s_e.unsqueeze(1), u_neg_pre),dim=-1)
        denoise = torch.mean(torch.sum(temp,axis=-1))




        n_e_ = n_hard_syth + n_easy_syth        
        hard_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_hard), axis=-1)  # [batch_size, K]
        easy_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_easy), axis=-1)  # [batch_size, K]
        syth_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e_), axis=-1)  # [batch_size, K]
        norm_scores = torch.sum(torch.mul(s_e.unsqueeze(dim=1), n_e), axis=-1)  # [batch_size, K]
        sns_loss = torch.mean(torch.log(1 + torch.exp(easy_scores - hard_scores).sum(dim=1)))
        dis_loss = distance + orth
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs]
        scores_false =  syth_scores - norm_scores

        indices = torch.max(scores + self.eps*scores_false, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        neg_embeddings = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices, :]
        

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1).squeeze(dim=1).sum(dim=-1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            neg_ego_embeddings,
            require_pow=self.require_pow,
        )


        loss = mf_loss + self.reg_weight * reg_loss + self.gamma * (sns_loss + dis_loss) + self.gamma * denoise
        # loss = mf_loss + self.gamma * (sns_loss)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

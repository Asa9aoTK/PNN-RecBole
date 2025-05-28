

import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class PNNP(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PNNP, self).__init__(config, dataset)

        self.history_item_id, _, self.history_item_len = dataset.history_item_matrix(
            max_history_len=config["history_len"]
        )
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_len = self.history_item_len.to(self.device)

        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.neg_seq_len = config["train_neg_sample_args"]["sample_num"]
        self.reg_weight = config["reg_weight"]
        self.history_len = torch.max(self.history_item_len, dim=0)


        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        self.item_mu = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.item_var = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        self.uncer_mlp = nn.Sequential(
            nn.Linear(2*self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        self.W1 = nn.Linear(self.embedding_size, 1)
        self.W2 = nn.Linear(self.embedding_size, self.embedding_size)
        

        self.score_buffer = defaultdict(lambda: deque(maxlen=5))
        
        self.reg_loss = EmbLoss()
        self.BPR = BPRLoss()
        self.apply(xavier_normal_initialization)
        self.item_mu.weight.data[0, :] = 0
        self.item_var.weight.data[0, :] = 0

    def wasserstein_score(self, user_mu, item_mu, item_var):
        mu_diff = torch.norm(user_mu - item_mu, p=2, dim=-1)
        trace_term = torch.sum(item_var, dim=-1)  # 简化计算
        return - (mu_diff.pow(2) + trace_term)

    def compute_uncertainty(self, user_e, item_e, pos_e):
        combined = torch.cat([user_e, item_e], dim=-1)
        e_temp = torch.sigmoid(self.uncer_mlp(combined))
        
        score_u = torch.sigmoid(torch.abs(self.wasserstein_score(user_e, item_e, self.item_var(item_e)) - 
                                       torch.mean(self.wasserstein_score(user_e, item_e, self.item_var(item_e)))))
        
        score_i = torch.sigmoid(1.0 / (torch.norm(pos_e - item_e, p=2, dim=-1) + 1e-8))
        
        key = (user_e.sum().item(), item_e.sum().item())  # 简化缓存key
        current_score = (score_u + score_i).mean().item()
        self.score_buffer[key].append(current_score)
        final_score = torch.tensor(np.mean(list(self.score_buffer[key])), device=self.device)
        
        return final_score * e_temp

    def forward(self, user, pos_item, history_item, history_len, neg_item_seq):
        user_e = self.user_emb(user)
        pos_mu = self.item_mu(pos_item)
        neg_mu = self.item_mu(neg_item_seq)
        
        neg_var = torch.sigmoid(self.compute_uncertainty(
            user_e.unsqueeze(1).expand(-1, neg_item_seq.size(1), -1).reshape(-1, self.embedding_size),
            neg_mu.view(-1, self.embedding_size),
            pos_mu.unsqueeze(1).expand(-1, neg_item_seq.size(1), -1).reshape(-1, self.embedding_size)
        )).view(neg_item_seq.shape[0], neg_item_seq.shape[1], self.embedding_size)
        
        pos_score = self.wasserstein_score(user_e, pos_mu, torch.zeros_like(pos_mu))
        neg_score = self.wasserstein_score(user_e.unsqueeze(1), neg_mu, neg_var)
        
        attn_weights = torch.softmax(torch.tanh(self.W2(pos_mu)) @ user_e.unsqueeze(-1), dim=0)
        lambda_val = torch.sigmoid(self.W1(torch.sum(attn_weights * pos_mu, dim=0)))
        
        bpr_loss = self.BPR(pos_score, neg_score.mean(dim=1))
        
        clamp_pos = pos_mu + torch.rand_like(pos_mu)*0.1
        clamp_neg = neg_mu.mean(dim=1) - torch.rand_like(neg_mu.mean(dim=1))*0.1
        constrain_loss = torch.norm(clamp_pos - clamp_neg, p=2, dim=-1).mean()
        uncer_loss = torch.sum(neg_var, dim=-1).mean()
        
        total_loss = (1 - lambda_val)*bpr_loss + lambda_val*(
            self.alpha*constrain_loss + 
            self.beta*uncer_loss +
            self.BPR(pos_score, neg_score.mean(dim=1)) +
            self.BPR(neg_score.mean(dim=1), neg_score.min(dim=1)[0])
        )
        
        reg_loss = self.reg_loss(user_e, pos_mu, neg_mu)
        return total_loss + self.reg_weight * reg_loss

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        neg_item_seq = neg_item.reshape((self.neg_seq_len, -1)).T
        user_number = len(user) // self.neg_seq_len
        
        user = user[:user_number]
        history_item = self.history_item_id[user]
        pos_item = pos_item[:user_number]
        history_len = self.history_item_len[user]

        return self.forward(user, pos_item, history_item, history_len, neg_item_seq)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]
        test_item = interaction[self.ITEM_ID]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, embedding_size]
        test_item_e = self.item_emb(test_item)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_cos = self.get_cos(UI_aggregation_e, test_item_e.unsqueeze(1))
        return UI_cos.squeeze(1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        history_item = self.history_item_id[user]
        history_len = self.history_item_len[user]

        # [user_num, embedding_size]
        user_e = self.user_emb(user)
        # [user_num, max_history_len, embedding_size]
        history_item_e = self.item_emb(history_item)

        # [user_num, embedding_size]
        UI_aggregation_e = self.get_UI_aggregation(user_e, history_item_e, history_len)

        UI_aggregation_e = F.normalize(UI_aggregation_e, dim=1)
        all_item_emb = self.item_emb.weight
        all_item_emb = F.normalize(all_item_emb, dim=1)
        UI_cos = torch.matmul(UI_aggregation_e, all_item_emb.T)
        return UI_cos

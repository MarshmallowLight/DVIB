# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


class VBPRA(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPRA, self).__init__(config, dataloader)

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  
        self.consistency_weight = config['consistency_weight']
        self.noise_level_net = NoiseLevelNetwork(input_dim=config['embedding_size']) 

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        p = Normal(0, 0.5) 
        item_embeddings = self.item_linear(self.item_raw_features)
        adaptive_noise_level = self.noise_level_net(item_embeddings) 
        sigma = adaptive_noise_level
        sigma = torch.clamp(sigma, min=1e-6) 
        noisy_item_embeddings = item_embeddings + adaptive_noise_level * torch.randn_like(item_embeddings) 
        q = Normal(item_embeddings, sigma) 
        vib_loss = kl_divergence(q, p).mean()  
        vib_loss *= 0.001 
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)
        noisy_item_embeddings = torch.cat((self.i_embedding, noisy_item_embeddings), -1) 


        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        noisy_item_e = F.dropout(noisy_item_embeddings, dropout) 
        return user_e, item_e, noisy_item_e, vib_loss 
    
    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean') 
        kl_qm = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean')  
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings, noisy_item_embeddings, vib_loss = self.forward() 
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        noisy_all_item_e = noisy_item_embeddings 
        pos_e = item_embeddings[pos_item, :]
        noisy_pos_e = noisy_item_embeddings[pos_item, :] 
        neg_e = item_embeddings[neg_item, :]
        noisy_neg_e = noisy_item_embeddings[neg_item, :] 
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        noisy_pos_item_score, noisy_neg_item_score = torch.mul(user_e, noisy_pos_e).sum(dim=1), torch.mul(user_e, noisy_neg_e).sum(dim=1) 
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        noisy_mf_loss = self.loss(noisy_pos_item_score, noisy_neg_item_score) 
        noisy_reg_loss = self.reg_loss(user_e, noisy_pos_e, noisy_neg_e) 
        loss = mf_loss + self.reg_weight * reg_loss
        # 1. general loss
        general_loss = loss + noisy_mf_loss + self.reg_weight * noisy_reg_loss 
        # 2. consistency loss
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        noisy_score = torch.matmul(user_e, noisy_all_item_e.transpose(0, 1))
        score = torch.softmax(score, dim=-1)  
        noisy_score = torch.softmax(noisy_score, dim=-1)  
        consistency_loss = self.js_divergence(score, noisy_score)
        loss = (1 - self.consistency_weight) * general_loss + self.consistency_weight * consistency_loss 
        loss += vib_loss 
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, a, b = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

class NoiseLevelNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(NoiseLevelNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        noise_level = self.sigmoid(self.fc3(x))
        return noise_level
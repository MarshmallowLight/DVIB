# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
paper:  Bootstrap Latent Representations for Multi-modal Recommendation
https://arxiv.org/abs/2207.05969
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal


class BM3A(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3A, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.consistency_weight = config['consistency_weight']
        self.noise_level_net = NoiseLevelNetwork(input_dim=config['feat_embed_dim']) 

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        original_weight = self.user_embedding.weight
        random_tensor = 0.01 * torch.randn_like(original_weight)
        new_weight = nn.Parameter(original_weight + random_tensor)
        self.user_embedding.weight = new_weight
        original_weight = self.item_id_embedding.weight
        random_tensor = 0.01 * torch.randn_like(original_weight)
        new_weight = nn.Parameter(original_weight + random_tensor)
        self.item_id_embedding.weight = new_weight

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
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

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h 
    
    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean')  
        kl_qm = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean')  
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        vib_loss_t, vib_loss_v = 0.0, 0.0 
        adaptive_noise_level_t, adaptive_noise_level_v = 0.0001, 0.0001 
        p = Normal(0, 0.5)
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
            adaptive_noise_level_t = self.noise_level_net(t_feat_online)
            sigma_t = adaptive_noise_level_t
            sigma_t = torch.clamp(sigma_t, min=1e-6)
            noisy_t_feat_online = t_feat_online + adaptive_noise_level_t * torch.randn_like(t_feat_online) 
            q = Normal(t_feat_online, sigma_t) 
            vib_loss_t = kl_divergence(q, p).mean()  
            vib_loss_t *= 0.001 
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)
            adaptive_noise_level_v = self.noise_level_net(v_feat_online)
            noisy_v_feat_online = v_feat_online + adaptive_noise_level_v * torch.randn_like(v_feat_online) 
            sigma_v = adaptive_noise_level_v 
            sigma_v = torch.clamp(sigma_v, min=1e-6)
            q = Normal(v_feat_online, sigma_v)
            vib_loss_v = kl_divergence(q, p).mean()
            vib_loss_v *= 0.001 

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)
                noisy_t_feat_target = noisy_t_feat_online.clone() 
                noisy_t_feat_target = F.dropout(noisy_t_feat_target, self.dropout) 

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)
                noisy_v_feat_target = noisy_v_feat_online.clone() 
                noisy_v_feat_target = F.dropout(noisy_v_feat_target, self.dropout) 

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
            noisy_t_feat_online = self.predictor(noisy_t_feat_online) 
            noisy_t_feat_online = noisy_t_feat_online[items, :] 
            noisy_t_feat_target = noisy_t_feat_target[items, :] 
            noisy_loss_t = 1 - cosine_similarity(noisy_t_feat_online, i_target.detach(), dim=-1).mean() 
            noisy_loss_tv = 1 - cosine_similarity(noisy_t_feat_online, noisy_t_feat_target.detach(), dim=-1).mean()  
        if self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()
            noisy_v_feat_online = self.predictor(noisy_v_feat_online) 
            noisy_v_feat_online = noisy_v_feat_online[items, :] 
            noisy_v_feat_target = noisy_v_feat_target[items, :] 
            noisy_loss_v = 1 - cosine_similarity(noisy_v_feat_online, i_target.detach(), dim=-1).mean() 
            noisy_loss_vt = 1 - cosine_similarity(noisy_v_feat_online, noisy_v_feat_target.detach(), dim=-1).mean() 

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        #1. general loss
        general_loss = (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) 
        general_loss += self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()
        general_loss += self.cl_weight * (noisy_loss_t + noisy_loss_v + noisy_loss_tv + noisy_loss_vt).mean() 

        # 2. consistency loss
        consistency_loss_t, consistency_loss_v = 0.0, 0.0
        if self.t_feat is not None:
            t_feat_online = torch.softmax(t_feat_online, dim=-1)
            t_feat_target = torch.softmax(t_feat_target, dim=-1) 
            noisy_t_feat_online = torch.softmax(noisy_t_feat_online, dim=-1)
            noisy_t_feat_target = torch.softmax(noisy_t_feat_target, dim=-1)
            consistency_loss_t = self.js_divergence(t_feat_online, noisy_t_feat_online)
            consistency_loss_t += self.js_divergence(t_feat_target, noisy_t_feat_target)
        if self.v_feat is not None:
            v_feat_online = torch.softmax(v_feat_online, dim=-1)
            v_feat_target = torch.softmax(v_feat_target, dim=-1) 
            noisy_v_feat_online = torch.softmax(noisy_v_feat_online, dim=-1)
            noisy_v_feat_target = torch.softmax(noisy_v_feat_target, dim=-1)
            consistency_loss_v = self.js_divergence(v_feat_online, noisy_v_feat_online)
            consistency_loss_v += self.js_divergence(v_feat_target, noisy_v_feat_target)
        
        consistency_loss = consistency_loss_t + consistency_loss_v 
        loss = (1 - self.consistency_weight) * general_loss + self.consistency_weight * consistency_loss 
        loss += vib_loss_t + vib_loss_v 

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui

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
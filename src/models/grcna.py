# coding: utf-8
# 
"""
Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback, MM 2020
"""
import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
#from SAGEConv import SAGEConv
#from GATConv import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
# from torch.utils.checkpoint import checkpoint
##########################################################################

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='mean', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr='add')#, **kwargs)
        self.self_loops = self_loops
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)


    def message(self,  x_i, x_j, size_i ,edge_index_i):
        #print(edge_index_i, x_i, x_j)
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)
        #print(self.alpha)
        #print(edge_index_i,size_i)
        # alpha = F.tanh(alpha)
        # self.alpha = F.leaky_relu(self.alpha)
        # alpha = torch.sigmoid(alpha)
        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)
        # Sample attention coefficients stochastically.
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j*self.alpha.view(-1,1)
        # return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        return aggr_out



class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_E = dim_E
        self.aggr_mode = aggr_mode
        self.has_act = has_act
        self.has_norm = has_norm
        self.id_embedding = nn.Parameter( nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_E))))
        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)         
        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

        if self.has_norm:
            x = F.normalize(x) 

        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector) 

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)
        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(self, features, num_user, num_item, dim_C, aggr_mode, num_routing, has_act, has_norm, is_word=False):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C))))
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word

        if is_word:
            self.word_tensor = torch.LongTensor(features).cuda()
            self.features = nn.Embedding(torch.max(features[1])+1, dim_C)
            nn.init.xavier_normal_(self.features.weight)

        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = nn.Linear(self.dim_feat, self.dim_C)
            #print('MLP weight',self.MLP.weight)
            nn.init.xavier_normal_(self.MLP.weight)
            #print(self.MLP.weight)

    def forward(self, edge_index):
        #print(self.features)
        features = F.leaky_relu(self.MLP(self.features))
        #print('features',features)
        
        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)
            #print(preference,features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            #print(x,edge_index)
            x_hat_1 = self.conv_embed_1(x, edge_index) 
            preference = preference + x_hat_1[:self.num_user]

            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)

        x_hat_1 = self.conv_embed_1(x, edge_index) 

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCNA(GeneralRecommender):
    def __init__(self,  config, dataset):
        super(GRCNA, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        dim_x = config['embedding_size']
        dim_C = config['latent_embedding']
        num_layer = config['n_layers']
        batch_size = config['train_batch_size']         
        self.aggr_mode = 'add'
        self.weight_mode = 'confid'
        self.fusion_mode = 'concat'
        has_id = True
        has_act= False
        has_norm= True
        is_word = False
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = config['reg_weight']
        self.dropout = 0
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        #self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0
        self.id_gcn = EGCN(num_user, num_item, dim_x, self.aggr_mode, has_act, has_norm)
        self.pruning = True
        self.noise_level = config['noise_level'] 
        self.consistency_weight = config['consistency_weight'] 
        self.noise_level_net = NoiseLevelNetwork(input_dim=128) 
        

        num_model = 0
        if self.v_feat is not None:
            self.v_gcn = CGCN(self.v_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm)
            num_model += 1
        
        if self.t_feat is not None:
            self.t_gcn = CGCN(self.t_feat, num_user, num_item, dim_C, self.aggr_mode, num_layer, has_act, has_norm, is_word)
            num_model += 1

        self.model_specific_conf = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, num_model))))

        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).to(self.device)
        self.noisy_result = self.result 
        
        
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))
    
    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(F.log_softmax(p, dim=-1), m, reduction='batchmean')
        kl_qm = F.kl_div(F.log_softmax(q, dim=-1), m, reduction='batchmean')  
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd

    def forward(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
        #print('edge_index: ', edge_index)

        if self.v_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            if torch.isnan(weight_v).any() or torch.isinf(weight_v).any():
                print("v_rep contains NaN or inf!")
            weight = weight_v
            content_rep = v_rep

        if self.t_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t   
                content_rep = t_rep
            else:
                content_rep = torch.cat((content_rep,t_rep),dim=1)
                weight = torch.cat((weight, weight_t), dim=1) 
        adaptive_noise_level = 0.0001 
        adaptive_noise_level = self.noise_level_net(content_rep) 
        noisy_content_rep = content_rep + adaptive_noise_level * torch.randn_like(content_rep) 
        p = Normal(0,0.5) 
        q = Normal(content_rep, adaptive_noise_level) 
        vib_loss = kl_divergence(q,p).mean()
        vib_loss *= 0.001

        if self.weight_mode == 'confid':
            confidence = torch.cat((self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim=0)
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
            

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)


        if self.fusion_mode == 'concat':
            representation = torch.cat((id_rep, content_rep), dim=1)
            noisy_representation = torch.cat((id_rep, noisy_content_rep), dim=1) 
        self.result = representation
        self.noisy_result = noisy_representation 

        return representation, noisy_representation, vib_loss 

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out, noisy_out, vib_loss = self.forward() 
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        noisy_user_score = noisy_out[user_tensor]
        noisy_item_score = noisy_out[item_tensor] 
        noisy_score = torch.sum(noisy_user_score * noisy_item_score, dim=1).view(-1, 2) 
        noisy_loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(noisy_score, self.weight)))) 
        reg_embedding_loss = (self.id_gcn.id_embedding[user_tensor]**2 + self.id_gcn.id_embedding[item_tensor]**2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        reg_content_loss = torch.zeros(1).cuda() 
        if self.v_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor]**2).mean()
        if self.t_feat is not None:            
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor]**2).mean()

        reg_confid_loss = (self.model_specific_conf**2).mean()
        
        reg_loss = reg_embedding_loss + reg_content_loss

        reg_loss = self.reg_weight * reg_loss
        # 1. general loss
        general_loss = loss + noisy_loss + reg_loss*2 
        # 2. consistency loss
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        noisy_user_tensor = self.noisy_result[:self.n_users] 
        noisy_item_tensor = self.noisy_result[self.n_users:] 
        noisy_temp_user_tensor = noisy_user_tensor[interaction[0], :] 
        noisy_score_matrix = torch.matmul(noisy_temp_user_tensor, noisy_item_tensor.t()) 
        scores = torch.softmax(score_matrix, dim=-1) 
        noisy_scores = torch.softmax(noisy_score_matrix, dim=-1)  
        consistency_loss = self.js_divergence(scores, noisy_scores) 
        final_loss = (1 - self.consistency_weight) * general_loss + self.consistency_weight * consistency_loss 
        return final_loss+vib_loss
        
    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

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
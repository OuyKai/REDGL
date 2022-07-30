import math, copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_neighbourhood(adj, max=32, min=2):
    # min
    knn_val, knn_ind = torch.topk(adj, min, dim=-1)
    min_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    # max
    knn_val, knn_ind = torch.topk(adj, max, dim=-1)
    max_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)

    weighted_adjacency_matrix = torch.where(adj > torch.FloatTensor([0.66])[0].cuda(), adj, torch.FloatTensor([0])[0].cuda())
    weighted_adjacency_matrix = torch.where(min_matrix == torch.FloatTensor([0])[0].cuda(), weighted_adjacency_matrix, min_matrix)
    weighted_adjacency_matrix = torch.where(max_matrix > torch.FloatTensor([0])[0].cuda(), weighted_adjacency_matrix, torch.FloatTensor([0])[0].cuda())

    return weighted_adjacency_matrix

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def PCALoss(x, coefficient=10):
    embedding = torch.nn.functional.normalize(x, dim=1)
    conv = torch.cov(embedding.T)
    on_diag = torch.diagonal(conv).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(conv).pow_(2).sum()
    # (seed 123)baby 100 / (seed 10)clothing 2 / (seed 10)sports 0.1
    return (on_diag + off_diag) / coefficient

def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class AttnLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0):
        super(AttnLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.activate = F.relu
        self.LN = nn.LayerNorm(hidden_dim, eps=0.001)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.reattn_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        feats, _ = attention(query, key, value)

        residual_feats= self.activate(self.reattn_linear(feats))
        if self.dropout is not None:
            residual_feats = self.dropout(residual_feats)
        feats = self.LN(residual_feats + feats)
        return feats.squeeze()

class MultiHeadedAttention(nn.Module):
    """
    Implements 'Multi-Head Attention' proposed in the paper.
    """

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args
from utility.utils import build_sim, build_neighbourhood, compute_normalized_laplacian, PCALoss, AttnLayer
args = parse_args()


class REDGL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, negative_slope=0.1, attn_dropout=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(n_items, self.embedding_dim)
        self.linearW = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.attn = AttnLayer(self.embedding_dim, dropout=attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_neighbourhood(image_adj)
        image_adj = compute_normalized_laplacian(image_adj)
        torch.save(image_adj, '../data/%s/%s-core/image_adj_%d.pt'%(args.dataset, args.core, args.topk))

        text_adj = build_sim(self.text_embedding.weight.detach())
        image_adj = build_neighbourhood(image_adj)
        text_adj = compute_normalized_laplacian(text_adj)
        torch.save(text_adj, '../data/%s/%s-core/text_adj_%d.pt'%(args.dataset, args.core, args.topk))

        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()
        
        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, adj, build_item_graph=False):
        pca_loss = PCALoss(self.user_embedding.weight, 2) + PCALoss(self.item_embedding.weight, 2) 
        image_feats = self.image_trs(self.image_embedding.weight).unsqueeze(1)
        text_feats = self.text_trs(self.text_embedding.weight).unsqueeze(1)

        feats = torch.cat([image_feats, text_feats, self.item_embedding.weight.unsqueeze(1)], dim=1)
        feats = self.attn(self.item_embedding.weight.unsqueeze(1), feats, feats)
        
        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            learned_adj = build_sim(feats.squeeze(1))
            learned_adj = build_neighbourhood(learned_adj)
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        h = feats
        for i in range(args.n_layers):
            h = self.leaky_relu(torch.mm(self.item_adj, self.linearW(h)))
        
        pca_loss += PCALoss(h, 128)

        if args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)            
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, pca_loss
        elif args.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings, pca_loss
        elif args.cf_model == 'mf':
                return self.user_embedding.weight, self.item_embedding.weight + F.normalize(h, p=2, dim=1), pca_loss

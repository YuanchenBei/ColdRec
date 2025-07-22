import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss, InfoNCE
from util.databuilder import TorchGraphInterface


# Following the source code process: https://github.com/Coder-Yu/SELFRec/blob/main/model/graph/SimGCL.py
class SimGCL(BaseColdStartTrainer):
    def __init__(self, config):
        super(SimGCL, self).__init__(config)
        self.n_layers = self.args.layers
        self.model = SimGCL_Encoder(self.args, self.data, self.emb_size, self.n_layers, self.device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.args.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                model.eval()
                self.user_emb, self.item_emb = self.model()
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(self.user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1 = self.model(perturbed=True)
        user_view_2, item_view_2 = self.model(perturbed=True)
        user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.args.tau)
        item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.args.tau)
        return user_cl_loss + item_cl_loss

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()
    
    def batch_predict(self, users):
        with torch.no_grad():
            users = self.data.get_user_id_list(users)
            users = torch.tensor(users, device=self.device)
            score = torch.matmul(self.user_emb[users], self.item_emb.transpose(0, 1))
            return score


class SimGCL_Encoder(nn.Module):
    def __init__(self, args, data, emb_size, n_layers, device):
        super(SimGCL_Encoder, self).__init__()
        self.args = args
        self.data = data
        self.eps = args.eps
        self.emb_size = emb_size
        self.device = device
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).to(self.device)
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss
from util.databuilder import TorchGraphInterface


class NGCF(BaseColdStartTrainer):
    def __init__(self, config):
        super(NGCF, self).__init__(config)
        self.n_layers = self.args.layers
        self.model = NGCF_Encoder(self.data, self.emb_size, self.n_layers, self.device)

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
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                self.user_emb, self.item_emb = model()
                if epoch % 1 == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        print("user_emb shape:", self.user_emb.shape)
        if self.args.save_emb:
            torch.save(self.user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

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


class NGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, device):
        super(NGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.device = device
        self.norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(data.norm_adj).to(self.device)

        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(data.user_num, emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(data.item_num, emb_size))),
        })

        self.W_gc = nn.ModuleList()
        self.W_bi = nn.ModuleList()
        for _ in range(n_layers):
            self.W_gc.append(nn.Linear(emb_size, emb_size))
            self.W_bi.append(nn.Linear(emb_size, emb_size))

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]

        for layer in range(self.layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            sum_embeddings = self.W_gc[layer](side_embeddings)
            bi_embeddings = self.W_bi[layer](ego_embeddings * side_embeddings)
            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

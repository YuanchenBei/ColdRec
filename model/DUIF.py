import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss


class DUIF(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(DUIF, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                   warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                   warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                   user_content=user_content, item_content=item_content)
        self.model = DUIF_Encoder(args, self.data, self.emb_size, device)

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
            return score.cpu().numpy()


class DUIF_Encoder(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(DUIF_Encoder, self).__init__()
        self.args = args
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()
        if self.args.cold_object == 'item':
            self.item_projector = nn.Linear(self.data.item_content_dim, self.latent_size, bias=False)
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_projector = nn.Linear(self.data.user_content_dim, self.latent_size, bias=False)
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        })
        return embedding_dict

    def forward(self):
        if self.args.cold_object == 'item':
            return self.embedding_dict['user_emb'], self.item_projector(self.item_content)
        else:
            return self.user_projector(self.user_content), self.embedding_dict['item_emb']

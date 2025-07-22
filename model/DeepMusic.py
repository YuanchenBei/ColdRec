import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, l2_reg_loss, mse_loss


class DeepMusic(BaseColdStartTrainer):
    def __init__(self, config):
        super(DeepMusic, self).__init__(config)
        self.model = DeepMusic_Encoder(self.args, self.data, self.emb_size, self.device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb = model()
                if self.args.cold_object == 'item':
                    pos_item_emb = rec_item_emb[pos_idx]
                    pos_item_content_emb = model.generate_item_emb(pos_idx)
                    batch_loss = mse_loss(pos_item_emb, pos_item_content_emb) + l2_reg_loss(self.reg, pos_item_content_emb)
                else:
                    pos_user_emb = rec_user_emb[user_idx]
                    pos_user_content_emb = model.generate_user_emb(user_idx)
                    batch_loss = mse_loss(pos_user_emb, pos_user_content_emb) + l2_reg_loss(self.reg, pos_user_content_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = self.model()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if self.args.cold_object == 'item':
                    cold_item_gen_emb = model.generate_item_emb(self.data.mapped_cold_item_idx)
                    self.item_emb.data[self.data.mapped_cold_item_idx] = cold_item_gen_emb
                else:
                    cold_user_gen_emb = model.generate_user_emb(self.data.mapped_cold_user_idx)
                    self.user_emb.data[self.data.mapped_cold_user_idx] = cold_user_gen_emb
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
            now_best_user_emb, now_best_item_emb = self.model.forward()
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            if self.args.cold_object == 'item':
                now_cold_item_gen_emb = self.model.generate_item_emb(self.data.mapped_cold_item_idx)
                self.best_item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_gen_emb
            else:
                now_cold_user_gen_emb = self.model.generate_user_emb(self.data.mapped_cold_user_idx)
                self.best_user_emb.data[self.data.mapped_cold_user_idx] = now_cold_user_gen_emb

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


class DeepMusic_Encoder(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(DeepMusic_Encoder, self).__init__()
        self.args = args
        self.data = data
        self.latent_size = emb_size
        self.device = device
        self.embedding_dict = self._init_model()
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        self.transformation = nn.Sequential(
            nn.Linear(self.content_dim, 2 * self.latent_size),
            nn.ReLU(),
            nn.Linear(2 * self.latent_size, self.latent_size),
        )
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
        })
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']

    def generate_item_emb(self, gen_idx):
        item_content_mapped_emb = self.transformation(self.item_content[gen_idx])
        return item_content_mapped_emb

    def generate_user_emb(self, gen_idx):
        now = self.user_content[gen_idx]
        user_content_mapped_emb = self.transformation(now)
        return user_content_mapped_emb

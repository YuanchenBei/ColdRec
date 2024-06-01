import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise_LARA
import torch.nn.functional as F


# Following the source code process: https://github.com/changfengsun/LARA
class LARA(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(LARA, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)

        self.model = LARA_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise_LARA(self.data, self.batch_size)):
                pos_user_idx, pos_item_idx, neg_user_idx, neg_item_idx = batch
                if self.args.cold_object == 'item':
                    real_user_emb = model.embedding_dict['user_emb'][pos_user_idx]
                    neg_user_emb = model.embedding_dict['user_emb'][neg_user_idx]
                    fake_user_emb = model.generator(pos_item_idx)
                    D_real, D_logit_real = model.discriminator(pos_item_idx, real_user_emb)
                    D_fake, D_logit_fake = model.discriminator(pos_item_idx, fake_user_emb)
                    D_counter, D_logit_counter = model.discriminator(neg_item_idx, neg_user_emb)
                else:
                    real_item_emb = model.embedding_dict['item_emb'][pos_item_idx]
                    neg_item_emb = model.embedding_dict['item_emb'][neg_item_idx]
                    fake_item_emb = model.generator(pos_user_idx)
                    D_real, D_logit_real = model.discriminator(pos_user_idx, real_item_emb)
                    D_fake, D_logit_fake = model.discriminator(pos_user_idx, fake_item_emb)
                    D_counter, D_logit_counter = model.discriminator(neg_user_idx, neg_item_emb)
                D_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D_logit_real, torch.ones_like(D_logit_real)))
                D_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D_logit_fake, torch.zeros_like(D_logit_fake)))
                D_loss_counter = torch.mean(F.binary_cross_entropy_with_logits(D_logit_counter, torch.zeros_like(D_logit_counter)))
                D_loss = D_loss_real + D_loss_fake + D_loss_counter
                G_loss = torch.mean(F.binary_cross_entropy_with_logits(D_logit_fake, torch.ones_like(D_logit_fake)))
                batch_loss = D_loss + G_loss
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
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')

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

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class LARA_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(LARA_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        self.G_Linear1 = nn.Linear(self.content_dim + self.latent_size, self.latent_size)
        self.G_Linear2 = nn.Linear(self.latent_size, self.latent_size)
        self.G_Linear3 = nn.Linear(self.latent_size, self.latent_size)
        self.D_Linear1 = nn.Linear(self.content_dim + self.latent_size, self.latent_size)
        self.D_Linear2 = nn.Linear(self.latent_size, self.latent_size)
        self.D_Linear3 = nn.Linear(self.latent_size, self.latent_size)
        self.embedding_dict = self._init_model()

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
        })
        return embedding_dict

    def forward(self):
        if self.args.cold_object == 'item':
            feature = self.item_content
            l1_outputs = torch.tanh(self.G_Linear1(torch.cat([feature, self.embedding_dict['item_emb']], dim=-1)))
            l2_outputs = torch.tanh(self.G_Linear2(l1_outputs))
            fake_emb = torch.tanh(self.G_Linear3(l2_outputs))
            return self.embedding_dict['user_emb'], fake_emb
        else:
            feature = self.user_content
            l1_outputs = torch.tanh(self.G_Linear1(torch.cat([feature, self.embedding_dict['user_emb']], dim=-1)))
            l2_outputs = torch.tanh(self.G_Linear2(l1_outputs))
            fake_emb = torch.tanh(self.G_Linear3(l2_outputs))
            return fake_emb, self.embedding_dict['item_emb']

    def generator(self, instance_id):
        if self.args.cold_object == 'item':
            feature = self.item_content[instance_id]
            feed_in = torch.cat([feature, self.embedding_dict['item_emb'][instance_id]], dim=-1)
        else:
            feature = self.user_content[instance_id]
            feed_in = torch.cat([feature, self.embedding_dict['user_emb'][instance_id]], dim=-1)
        l1_outputs = torch.tanh(self.G_Linear1(feed_in))
        l2_outputs = torch.tanh(self.G_Linear2(l1_outputs))
        fake_instance = torch.tanh(self.G_Linear3(l2_outputs))
        return fake_instance

    def discriminator(self, instance_id, ui_emb):
        if self.args.cold_object == 'item':
            feature = self.item_content[instance_id]
        else:
            feature = self.user_content[instance_id]
        emb = torch.cat([feature, ui_emb], dim=-1)
        l1_outputs = torch.tanh(self.D_Linear1(emb))
        l2_outputs = torch.tanh(self.D_Linear2(l1_outputs))
        D_logit = self.D_Linear3(l2_outputs)
        D_prob = torch.sigmoid(D_logit)
        return D_prob, D_logit

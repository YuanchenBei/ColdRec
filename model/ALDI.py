import numpy as np
import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss
import torch.nn.functional as F


# Following official: https://github.com/zfnWong/ALDI (cold_start/ALDI.py)


def _aldi_item_frequency(data):
    """Match official main.py: per-item sum of 1/|user neighborhood| over users linked to the item."""
    freq = np.ones(data.item_num, dtype=np.float32)
    for item, user_ratings in data.training_set_i.items():
        mid = data.item[item]
        acc = 0.0
        for u in user_ratings:
            nu = len(data.training_set_u[u])
            acc += 1.0 / float(max(nu, 1))
        freq[mid] = acc
    return freq


class ALDI(BaseColdStartTrainer):
    def __init__(self, config):
        super(ALDI, self).__init__(config)
        if self.args.cold_object == 'user':
            raise Exception(
                'Cold user is not supported in ALDI due to its specific design for item cold-start problem.'
            )
        self.model = ALDI_Learner(self.args, self.data, self.emb_size, self.device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(_aldi_mapper_param_groups(model, self.reg), lr=self.lr)
        self.timer(start=True)
        epoch = -1
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx_t = torch.tensor(user_idx, dtype=torch.long, device=self.device)
                pos_idx_t = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                neg_idx_t = torch.tensor(neg_idx, dtype=torch.long, device=self.device)

                user_emb, pos_item_emb, neg_item_emb, gen_user_emb, gen_pos_item_emb, gen_neg_item_emb = (
                    model.get_training_embs(user_idx_t, pos_idx_t, neg_idx_t)
                )
                basic_loss = bpr_loss(gen_user_emb, gen_pos_item_emb, gen_neg_item_emb)
                teacher_pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
                teacher_neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
                teacher_rank_distance = teacher_pos_score - teacher_neg_score
                student_pos_score = torch.mul(gen_user_emb, gen_pos_item_emb).sum(dim=1)
                student_neg_score = torch.mul(gen_user_emb, gen_neg_item_emb).sum(dim=1)
                student_rank_distance = student_pos_score - student_neg_score

                rating_dist_loss = self.args.gamma * torch.mean(
                    torch.abs(teacher_pos_score - student_pos_score)
                    + torch.abs(teacher_neg_score - student_neg_score)
                )

                pos_w = model.pos_item_weights[pos_idx_t]

                rank_per = F.binary_cross_entropy_with_logits(
                    student_rank_distance, torch.sigmoid(teacher_rank_distance), reduction='none'
                )
                ranking_dist_loss = self.args.alpha * (pos_w * rank_per).mean()

                student_ii_logit = (gen_pos_item_emb * gen_pos_item_emb).sum(dim=1)
                student_ij_logit = (gen_pos_item_emb.matmul(gen_neg_item_emb.t())).mean(dim=1)
                student_iden_distance = student_ii_logit - student_ij_logit

                teacher_ii_logit = (pos_item_emb * pos_item_emb).sum(dim=1)
                teacher_ij_logit = (pos_item_emb.matmul(neg_item_emb.t())).mean(dim=1)
                teacher_iden_distance = teacher_ii_logit - teacher_ij_logit
                iden_per = F.binary_cross_entropy_with_logits(
                    student_iden_distance, torch.sigmoid(teacher_iden_distance), reduction='none'
                )
                iden_dist_loss = self.args.beta * (pos_w * iden_per).mean()

                batch_loss = basic_loss + rating_dist_loss + ranking_dist_loss + iden_dist_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                warm_user_emb, now_item_emb = self.model()
                cold_user_emb = self.model.get_generated_user_embs()
                cold_item_gen_emb = self.model.get_generated_item_embs(self.data.mapped_cold_item_idx)
                self.warm_user_emb = warm_user_emb.clone()
                self.cold_user_emb = cold_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_item_gen_emb
                if epoch % self.eval_every == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.epochs_ran = (epoch + 1) if self.maxEpoch > 0 else 0
        self.timer(start=False)
        model.eval()
        self.warm_user_emb, self.cold_user_emb, self.item_emb = (
            self.best_warm_user_emb,
            self.best_cold_user_emb,
            self.best_item_emb,
        )
        if self.args.save_emb:
            torch.save(
                self.warm_user_emb,
                f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_warm_user_emb.pt',
            )
            torch.save(
                self.cold_user_emb,
                f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_cold_user_emb.pt',
            )
            torch.save(
                self.item_emb,
                f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt',
            )

    def save(self):
        with torch.no_grad():
            self.model.eval()
            warm_best_user_emb, now_best_item_emb = self.model.forward()
            cold_best_user_emb = self.model.get_generated_user_embs()
            self.best_warm_user_emb = warm_best_user_emb.clone()
            self.best_cold_user_emb = cold_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            now_cold_item_gen_emb = self.model.get_generated_item_embs(self.data.mapped_cold_item_idx)
            self.best_item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_gen_emb

    def predict(self, u):
        score = torch.zeros(self.data.item_num, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score[self.data.mapped_warm_item_idx] = torch.matmul(
                self.warm_user_emb[u], self.item_emb[self.data.mapped_warm_item_idx].transpose(0, 1)
            )
            score[self.data.mapped_cold_item_idx] = torch.matmul(
                self.cold_user_emb[u], self.item_emb[self.data.mapped_cold_item_idx].transpose(0, 1)
            )
            return score.cpu().numpy()

    def batch_predict(self, users):
        score = torch.zeros(len(users), self.data.item_num, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            users = self.data.get_user_id_list(users)
            users = torch.tensor(users, device=self.device)
            score[:, self.data.mapped_warm_item_idx] = torch.matmul(
                self.warm_user_emb[users], self.item_emb[self.data.mapped_warm_item_idx].transpose(0, 1)
            )
            score[:, self.data.mapped_cold_item_idx] = torch.matmul(
                self.cold_user_emb[users], self.item_emb[self.data.mapped_cold_item_idx].transpose(0, 1)
            )
            return score


def _trunc_normal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _aldi_mapper_param_groups(model, weight_decay):
    """L2 only on Linear weights/biases (official: regularizer on FC vars; not on batch-norm scale/shift)."""
    decay, no_decay = [], []
    for tower in (model.user_tower, model.item_tower):
        for mod in tower.modules():
            if isinstance(mod, nn.Linear):
                decay.append(mod.weight)
                if mod.bias is not None:
                    decay.append(mod.bias)
            elif isinstance(mod, nn.BatchNorm1d):
                if mod.affine:
                    if mod.weight is not None:
                        no_decay.append(mod.weight)
                    if mod.bias is not None:
                        no_decay.append(mod.bias)
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
    ]


class ALDITower(nn.Module):
    """
    Official ALDI mapper: Linear -> BatchNorm -> tanh -> Linear (no activation on output).
    Item and user use separate module instances (separate weights).
    """

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.apply(_trunc_normal_init)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn(h)
        h = torch.tanh(h)
        return self.fc2(h)


class ALDI_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(ALDI_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        hidden = int(getattr(args, 'aldi_hidden', 200))
        self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(
            device
        )
        self.user_tower = ALDITower(self.latent_size, hidden, self.latent_size)
        self.item_tower = ALDITower(self.data.item_content_dim, hidden, self.latent_size)
        self.embedding_dict = self._init_model()

        train_n = max(len(data.training_data), 1)
        item_num = float(max(data.item_num, 1))
        user_num = float(max(data.user_num, 1))
        x_expect = (train_n / item_num) * (1.0 / max(train_n / user_num, 1e-12))
        self.freq_coef_a = float(args.freq_coef_M) / float(x_expect)

        raw_freq = _aldi_item_frequency(data)
        item_freq = torch.tensor(raw_freq, dtype=torch.float32, device=device)
        cap = float(np.tanh(float(args.freq_coef_M)))
        if int(getattr(args, 'tws', 0)):
            tw = torch.clamp(torch.tanh(self.freq_coef_a * item_freq), 0.0, cap)
        else:
            tw = torch.ones_like(item_freq)
        self.register_buffer('pos_item_weights', tw)

    def _init_model(self):
        embedding_dict = nn.ParameterDict(
            {
                'user_emb': torch.load(
                    f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt',
                    map_location='cpu',
                ),
                'item_emb': torch.load(
                    f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt',
                    map_location='cpu',
                ),
            }
        )
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict

    def get_training_embs(self, uid, pos_iid, neg_iid):
        user_emb = self.embedding_dict['user_emb'][uid]
        pos_item_emb = self.embedding_dict['item_emb'][pos_iid]
        neg_item_emb = self.embedding_dict['item_emb'][neg_iid]
        gen_user_emb = self.user_tower(user_emb)
        gen_pos_item_emb = self.item_tower(self.item_content[pos_iid])
        gen_neg_item_emb = self.item_tower(self.item_content[neg_iid])
        return user_emb, pos_item_emb, neg_item_emb, gen_user_emb, gen_pos_item_emb, gen_neg_item_emb

    def forward(self):
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        return user_emb, item_emb

    def get_generated_user_embs(self):
        return self.user_tower(self.embedding_dict['user_emb'])

    def get_generated_item_embs(self, gen_idx):
        if isinstance(gen_idx, torch.Tensor):
            idx = gen_idx
        else:
            idx = torch.tensor(gen_idx, dtype=torch.long, device=self.device)
        return self.item_tower(self.item_content[idx])

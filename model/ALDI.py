import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss
import torch.nn.functional as F


# Following the source code process: https://github.com/zfnWong/ALDI
class ALDI(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(ALDI, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)
        if self.args.cold_object == 'user':
            raise Exception('Cold user is not supported in ALDI due to its specific design for item cold-start problem.')
        self.model = ALDI_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb, gen_user_emb, gen_pos_item_emb, gen_neg_item_emb = model.get_training_embs(user_idx, pos_idx, neg_idx)
                basic_loss = bpr_loss(gen_user_emb, gen_pos_item_emb, gen_neg_item_emb)
                teacher_pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
                teacher_neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
                teacher_rank_distance = teacher_pos_score - teacher_neg_score
                student_pos_score = torch.mul(gen_user_emb, gen_pos_item_emb).sum(dim=1)
                student_neg_score = torch.mul(gen_user_emb, gen_neg_item_emb).sum(dim=1)
                student_rank_distance = student_pos_score - student_neg_score

                # Rating Distribution Aligning
                rating_dist_loss = self.args.gamma * torch.mean(torch.abs(teacher_pos_score - student_pos_score)+
                                                                torch.abs(teacher_neg_score - student_neg_score))
                # Ranking Aligning
                ranking_dist_loss = self.args.alpha * torch.mean(F.binary_cross_entropy_with_logits(student_rank_distance,
                                                                                  torch.sigmoid(teacher_rank_distance)))
                # Identity Distribution Aligning
                student_ii_logit = (gen_pos_item_emb * gen_pos_item_emb).sum(dim=1)
                student_ij_logit = (gen_pos_item_emb.matmul(gen_neg_item_emb.t())).mean(dim=1)
                student_iden_distance = student_ii_logit - student_ij_logit

                teacher_ii_logit = (pos_item_emb * pos_item_emb).sum(dim=1)
                teacher_ij_logit = (pos_item_emb.matmul(neg_item_emb.t())).mean(dim=1)
                teacher_iden_distance = teacher_ii_logit - teacher_ij_logit
                iden_dist_loss = self.args.beta * torch.mean(F.binary_cross_entropy_with_logits(student_iden_distance,
                                                                              torch.sigmoid(teacher_iden_distance)))

                batch_loss = basic_loss + rating_dist_loss + ranking_dist_loss + iden_dist_loss + \
                             l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb, gen_user_emb,
                                         gen_pos_item_emb, gen_neg_item_emb) / self.batch_size
                # Backward and optimize
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
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')

        model.eval()
        self.warm_user_emb, self.cold_user_emb, self.item_emb = self.best_warm_user_emb, self.best_cold_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(self.warm_user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_warm_user_emb.pt")
            torch.save(self.cold_user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_cold_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def save(self):
        with torch.no_grad():
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
            score[self.data.mapped_warm_item_idx] = torch.matmul(self.warm_user_emb[u], self.item_emb[self.data.mapped_warm_item_idx].transpose(0, 1))
            score[self.data.mapped_cold_item_idx] = torch.matmul(self.cold_user_emb[u], self.item_emb[self.data.mapped_cold_item_idx].transpose(0, 1))
            return score.cpu().numpy()


class ALDI_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(ALDI_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        self.user_emb_mapper = dense_batch_fc_tanh(self.latent_size, 2 * self.latent_size, self.latent_size)
        self.item_emb_mapper = dense_batch_fc_tanh(self.data.item_content_dim, 2 * self.latent_size, self.latent_size)
        self.embedding_dict = self._init_model()

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
        })
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict

    def get_training_embs(self, uid, iid, nid):
        user_emb = self.embedding_dict['user_emb'][uid]
        pos_item_emb = self.embedding_dict['item_emb'][iid]
        neg_item_emb = self.embedding_dict['item_emb'][nid]
        gen_user_emb = self.user_emb_mapper(user_emb)
        gen_pos_item_emb = self.item_emb_mapper(self.item_content[iid])
        gen_neg_item_emb = self.item_emb_mapper(self.item_content[nid])
        return user_emb, pos_item_emb, neg_item_emb, gen_user_emb, gen_pos_item_emb, gen_neg_item_emb

    def forward(self):
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        return user_emb, item_emb

    def get_generated_user_embs(self):
        return self.user_emb_mapper(self.embedding_dict['user_emb'])

    def get_generated_item_embs(self, gen_idx):
        return self.item_emb_mapper(self.item_content[gen_idx])


class dense_batch_fc_tanh(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(dense_batch_fc_tanh, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)

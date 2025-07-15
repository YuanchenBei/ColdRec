import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise
import torch.nn.functional as F


# Following the source code process: https://github.com/duxy-me/MTPR/blob/master/vbpr.py
class VBPR(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content, item_content):
        super(VBPR, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)
        self.model = VBPR_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        optimizer = torch.optim.Adagrad([self.model.P.weight, self.model.PQ2.weight, self.model.Q.weight], lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam([self.model.W], lr=lr2, weight_decay=0)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx, pos_idx, neg_idx = torch.LongTensor(user_idx).to(self.device), torch.LongTensor(pos_idx).to(self.device), torch.LongTensor(neg_idx).to(self.device)
                batch_loss = self.model.bpr_training(user_idx, pos_idx, neg_idx) + self.model.regs(user_idx, pos_idx, neg_idx)
                # Backward and optimize
                optimizer.zero_grad()
                optimizer2.zero_grad()
                batch_loss.backward()
                optimizer.step()
                optimizer2.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                model.eval()
                self.user_emb_main, self.item_emb_main, self.user_emb_aux, self.item_emb_aux, self.w_value = self.model()
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.timer(start=False)
        model.eval()
        self.user_emb_main, self.item_emb_main, self.user_emb_aux, self.item_emb_aux, self.w_value = self.best_user_emb_main, self.best_item_emb_main, self.best_user_emb_aux, self.best_item_emb_aux, self.best_w
        if self.args.save_emb:
            torch.save(self.user_emb_main, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb_main_P.pt")
            torch.save(self.item_emb_main, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb_main_Q.pt")
            torch.save(self.user_emb_aux, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb_aux.pt")
            torch.save(self.item_emb_aux, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb_aux.pt")
            torch.save(self.w_value, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_W.pt")

    def save(self):
        with torch.no_grad():
            self.best_user_emb_main, self.best_item_emb_main, self.best_user_emb_aux, self.best_item_emb_aux, self.best_w = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score1 = torch.matmul(self.user_emb_main[u], self.item_emb_main.transpose(0, 1))
            score2 = torch.matmul(self.user_emb_aux[u], self.item_emb_aux.transpose(0, 1))
            score = score1 + score2
            return score.cpu().numpy()
    
    def batch_predict(self, users):
        with torch.no_grad():
            users = self.data.get_user_id_list(users)
            users = torch.tensor(users, device=self.device)
            score1 = torch.matmul(self.user_emb_main[users], self.item_emb_main.transpose(0, 1))
            score2 = torch.matmul(self.user_emb_aux[users], self.item_emb_aux.transpose(0, 1))
            score = score1 + score2
            return score.cpu().numpy()


class VBPR_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(VBPR_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        p_weight = torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu').data
        q_weight = torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu').data
        self.P = torch.nn.Embedding(self.data.user_num, self.latent_size)
        self.P.weight.data.copy_(p_weight)
        self.P.weight.requires_grad = True

        self.Q = torch.nn.Embedding(self.data.item_num, self.latent_size)
        self.Q.weight.data.copy_(q_weight)
        self.Q.weight.requires_grad = True

        if self.args.cold_object == 'item':
            self.PQ2 = torch.nn.Embedding(self.data.user_num, self.latent_size)
            initializer(self.PQ2.weight.data)
            self.PQ2.weight.requires_grad = True
            self.W = initializer(torch.randn(self.data.item_content_dim, self.latent_size, dtype=torch.float32).to(self.device))
            self.W.requires_grad = True
        else:
            self.PQ2 = torch.nn.Embedding(self.data.item_num, self.latent_size)
            initializer(self.PQ2.weight.data)
            self.PQ2.weight.requires_grad = True
            self.W = initializer(torch.randn(self.data.user_content_dim, self.latent_size, dtype=torch.float32).to(self.device))
            self.W.requires_grad = True

    def forward(self):
        user_emb_main = self.P.weight
        item_emb_main = self.Q.weight
        if self.args.cold_object == 'item':
            user_emb_aux = self.PQ2.weight
            item_emb_aux = torch.mm(self.item_content, self.W)
        else:
            user_emb_aux = torch.mm(self.user_content, self.W)
            item_emb_aux = self.PQ2.weight
        return user_emb_main, item_emb_main, user_emb_aux, item_emb_aux, self.W

    def predict(self, uid, iid):
        if self.args.cold_object == 'item':
            p1 = torch.sum(self.P(uid) * self.Q(iid), dim=1)
            p2 = torch.sum(self.PQ2(uid) * torch.mm(self.item_content[iid], self.W), dim=1)
        else:
            p1 = torch.sum(self.P(uid) * self.Q(iid), dim=1)
            p2 = torch.sum(torch.mm(self.user_content[uid], self.W) * self.PQ2(iid), dim=1)
        return p1 + p2

    def bpr_training(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx

        if self.args.cold_object == 'item':
            p1 = self.P(uid)
            q = self.Q(iid)
            qn = self.Q(niid)
            p2 = self.PQ2(uid)
            w = self.W
            emb_regs = torch.sum(p1 * p1) + torch.sum(p2 * p2) + torch.sum(q * q) + torch.sum(qn * qn)
            ctx_regs = torch.sum(w * w)
        else:
            p1 = self.P(uid)
            q = self.Q(iid)
            qn = self.Q(niid)
            w = self.W
            p2 = self.PQ2(iid)
            p3 = self.PQ2(niid)
            emb_regs = torch.sum(p1 * p1) + torch.sum(p2 * p2) + torch.sum(p3 * p3) + torch.sum(q * q) + torch.sum(qn * qn)
            ctx_regs = torch.sum(w * w)

        return wd1 * emb_regs + wd2 * ctx_regs

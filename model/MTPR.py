import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise
import torch.nn.functional as F


# Following the source code process: https://github.com/duxy-me/MTPR/blob/master/mtpr.py
class MTPR(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(MTPR, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)
        self.model = MTPR_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        lr3, wd3 = self.args.p_proj
        optimizer = torch.optim.Adagrad([self.model.P.weight, self.model.Q.weight], lr=lr1)
        optimizer2 = torch.optim.Adam([self.model.W, self.model.weu], lr=lr2)
        optimizer3 = torch.optim.Adam([self.model.wei], lr=lr3)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx, pos_idx, neg_idx = torch.LongTensor(user_idx).to(self.device), torch.LongTensor(pos_idx).to(self.device), torch.LongTensor(neg_idx).to(self.device)
                batch_loss = self.model.mtl_loss(user_idx,pos_idx, neg_idx) + self.model.regs(user_idx, pos_idx, neg_idx)
                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                batch_loss.backward()
                optimizer.step()
                optimizer2.step()
                optimizer3.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                model.eval()
                self.user_emb, self.item_emb = self.model()
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
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class MTPR_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(MTPR_Learner, self).__init__()
        self.args = args
        self.data = data
        self.latent_size = emb_size
        self.device = device
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        if self.args.cold_object == 'item':
            self.P = torch.nn.Embedding(self.data.user_num, self.latent_size * 2).to(self.device)
        else:
            self.P = torch.nn.Embedding(self.data.user_num, self.latent_size).to(self.device)
        initializer(self.P.weight)
        self.P.weight.requires_grad = True

        if self.args.cold_object == 'item':
            self.Q = torch.nn.Embedding(self.data.item_num, self.latent_size).to(self.device)
        else:
            self.Q = torch.nn.Embedding(self.data.item_num, self.latent_size * 2).to(self.device)
        initializer(self.Q.weight)
        self.Q.weight.requires_grad = True

        if self.args.cold_object == 'item':
            self.W = initializer(torch.randn(self.data.item_content_dim, self.latent_size, dtype=torch.float32).to(self.device))
        else:
            self.W = initializer(torch.randn(self.data.user_content_dim, self.latent_size, dtype=torch.float32).to(self.device))
        self.W.requires_grad = True

        self.weu = initializer(torch.randn(self.latent_size * 2, self.latent_size, dtype=torch.float32).to(self.device))
        self.weu.requires_grad = True
        self.wei = initializer(torch.randn(self.latent_size * 2, self.latent_size, dtype=torch.float32).to(self.device))
        self.wei.requires_grad = True

    def forward(self):
        if self.args.cold_object == 'item':
            user_emb = self.trf(self.P.weight, self.weu)
            item_emb = self.trf(torch.cat((self.Q.weight, torch.mm(self.item_content, self.W)), dim=1), self.wei)
        else:
            user_emb = self.trf(torch.cat((self.P.weight, torch.mm(self.user_content, self.W)), dim=1), self.weu)
            item_emb = self.trf(self.Q.weight, self.wei)
        return user_emb, item_emb

    def trf(self, emb, theta):
        return torch.mm(emb, theta)

    def fimg_item(self, iid):  # normal representation
        return torch.cat((self.Q(iid), torch.mm(self.item_content[iid], self.W)), dim=1)

    def zimg_item(self, iid):  # conterfactual representation
        fzero = torch.zeros_like(self.Q(iid))
        return torch.cat((fzero, torch.mm(self.item_content[iid], self.W)), dim=1)

    def fimg_user(self, uid):
        return torch.cat((self.P(uid), torch.mm(self.user_content[uid], self.W)), dim=1)

    def zimg_user(self, uid):
        fzero = torch.zeros_like(self.P(uid))
        return torch.cat((fzero, torch.mm(self.user_content[uid], self.W)), dim=1)

    def predict(self, uid, iid):
        if self.args.cold_object == 'item':
            return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.fimg_item(iid), self.wei), dim=1)
        else:
            return torch.sum(self.trf(self.fimg_user(uid), self.weu) * self.trf(self.Q(iid), self.wei), dim=1)

    def predict_z(self, uid, iid):
        if self.args.cold_object == 'item':
            return torch.sum(self.trf(self.P(uid), self.weu) * self.trf(self.zimg_item(iid), self.wei), dim=1)
        else:
            return torch.sum(self.trf(self.zimg_user(uid), self.weu) * self.trf(self.Q(iid), self.wei), dim=1)

    def bpr_loss_i(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_f(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_if(self, uid, iid, niid):
        pred_p = self.predict(uid, iid)
        pred_n = self.predict_z(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    def bpr_loss_fi(self, uid, iid, niid):
        pred_p = self.predict_z(uid, iid)
        pred_n = self.predict(uid, niid)
        result = pred_p - pred_n
        loss = torch.sum(F.softplus(-result))
        return loss

    # multi-task learning
    def mtl_loss(self, uid, iid, niid):
        aloss = 0
        aloss += self.bpr_loss_i(uid, iid, niid) + self.bpr_loss_f(uid, iid, niid)  # two crucial task
        aloss += self.bpr_loss_if(uid, iid, niid) + self.bpr_loss_fi(uid, iid, niid)  # two constraints
        return aloss

    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_ctx
        lr3, wd3 = self.args.p_proj

        p = self.P(uid)
        q = self.Q(iid)
        qn = self.Q(niid)
        w = self.W
        weu = self.weu
        wei = self.wei
        emb_regs = torch.sum(p * p) + torch.sum(q * q) + torch.sum(qn * qn)
        ctx_regs = torch.sum(w * w) + torch.sum(weu * weu)
        proj_regs = torch.sum(wei * wei)

        return wd1 * emb_regs + wd2 * ctx_regs + wd3 * proj_regs

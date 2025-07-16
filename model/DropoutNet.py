import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise


class DropoutNet(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(DropoutNet, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)

        self.model = DropoutNet_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = torch.nn.MSELoss()
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                pos_pair_score = model.pair_score(user_idx, pos_idx)
                neg_pair_score = model.pair_score(user_idx, neg_idx)
                batch_score = torch.cat((pos_pair_score, neg_pair_score), dim=0)
                pos_pair_pred = model.deepcf_forward(user_idx, pos_idx, is_drop=True)
                neg_pair_pred = model.deepcf_forward(user_idx, neg_idx, is_drop=True)
                batch_pred = torch.cat((pos_pair_pred, neg_pair_pred), dim=0)
                batch_loss = crit(batch_pred, batch_score)
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


class DropoutNet_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(DropoutNet_Learner, self).__init__()
        self.args = args
        self.emb_size = emb_size
        self.data = data
        self.device = device
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self.embedding_dict = self._init_model()
        if self.args.cold_object == 'item':
            self.deepcf_encoder = get_model(emb_size, 0, self.data.item_content_dim, [200, 100], emb_size)
        else:
            self.deepcf_encoder = get_model(emb_size, self.data.user_content_dim, 0, [200, 100], emb_size)

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
        })
        return embedding_dict

    def pair_score(self, uid, iid):
        user_emb = self.embedding_dict['user_emb'][uid]
        item_emb = self.embedding_dict['item_emb'][iid]
        return torch.sum(user_emb * item_emb, dim=1)

    def deepcf_forward(self, uid, iid, is_drop=False):
        user_emb = self.embedding_dict['user_emb'][uid]
        item_emb = self.embedding_dict['item_emb'][iid]
        if self.args.cold_object == 'item':
            if is_drop:
                num_rows_to_zero = int(item_emb.shape[0]*self.args.n_dropout)
                indices_to_zero = torch.randperm(item_emb.size(0))[:num_rows_to_zero]
                item_emb[indices_to_zero] = torch.zeros_like(item_emb[indices_to_zero]).to(self.device)
            item_content = self.item_content[iid]
            pred = self.deepcf_encoder(user_emb, item_emb, None, item_content)
        else:
            if is_drop:
                num_rows_to_zero = int(user_emb.shape[0]*self.args.n_dropout)
                indices_to_zero = torch.randperm(user_emb.size(0))[:num_rows_to_zero]
                user_emb[indices_to_zero] = torch.zeros_like(user_emb[indices_to_zero]).to(self.device)
            user_content = self.user_content[uid]
            pred = self.deepcf_encoder(user_emb, item_emb, user_content, None)
        return pred

    def forward(self):
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        if self.args.cold_object == 'item':
            item_content = self.item_content
            u_infer_emb, i_infer_emb = self.deepcf_encoder.encode(user_emb, item_emb, None, item_content)
        else:
            user_content = self.user_content
            u_infer_emb, i_infer_emb = self.deepcf_encoder.encode(user_emb, item_emb, user_content, None)
        return u_infer_emb, i_infer_emb


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


@torch.no_grad()
def init_weights(net):
    if type(net) == nn.Linear:
        truncated_normal_(net.weight, std=0.01)
        if net.bias is not None:
            torch.nn.init.constant_(net.bias, 0)


def get_model(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
    encoder = DeepCF(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out)
    encoder.apply(init_weights)
    return encoder


class DeepCF(nn.Module):
    """
    main model class implementing DeepCF
    also stores states for fast candidate generation
    latent_rank_in: rank of preference model input
    user_content_rank: rank of user content input
    item_content_rank: rank of item content input
    model_select: array of number of hidden unit,
        i.e. [200,100] indicate two hidden layer with 200 units followed by 100 units
    rank_out: rank of latent model output
    """

    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
        super(DeepCF, self).__init__()
        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out

        u_dim = self.rank_in + self.phi_u_dim if self.phi_u_dim > 0 else self.rank_in
        v_dim = self.rank_in + self.phi_v_dim if self.phi_v_dim > 0 else self.rank_in

        u_dims = [u_dim] + self.model_select
        v_dims = [v_dim] + self.model_select
        self.u_layers = nn.ModuleList(TanHBlock(u_dims[i], u_dims[i + 1]) for i in range(len(u_dims) - 1))
        self.v_layers = nn.ModuleList(TanHBlock(v_dims[i], v_dims[i + 1]) for i in range(len(v_dims) - 1))

        self.u_emb = nn.Linear(u_dims[-1], self.rank_out)
        self.v_emb = nn.Linear(v_dims[-1], self.rank_out)

    def encode(self, Uin, Vin, Ucontent, Vcontent):

        if self.phi_u_dim > 0:
            u_concat = torch.cat((Uin, Ucontent), 1)
        else:
            u_concat = Uin

        if self.phi_v_dim > 0:
            v_concat = torch.cat((Vin, Vcontent), 1)
        else:
            v_concat = Vin

        u_out = u_concat
        for layer in self.u_layers:
            u_out = layer(u_out)
        U_embedding = self.u_emb(u_out)

        v_out = v_concat
        for layer in self.v_layers:
            v_out = layer(v_out)
        V_embedding = self.v_emb(v_out)
        return U_embedding, V_embedding

    def forward(self, Uin, Vin, Ucontent, Vcontent):
        U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
        preds = U_embedding * V_embedding
        preds = torch.sum(preds, 1)
        return preds


class TanHBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TanHBlock, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(
            num_features=dim_out,
            momentum=0.01,
            eps=0.001
        )

    def forward(self, x):
        out = self.layer(x)
        out = self.bn(out)
        out = torch.tanh(out)
        return out

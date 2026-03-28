"""
M2VAE (hierarchical MoE + VAE) for item cold-start.

Training loss follows ML-20M model_hi_moe.py train(): additive KLD + recon + rank + contrast + decouple;
prior uses fc2([a_v, c_v]). No explicit L2 in loss (use Adam weight_decay).
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise_CCFCRec


def calc_kl_divergence(mu0, logvar0, mu1=None, logvar1=None):
    """KL as in model_hi_moe.py: sum over all elements, then mean (scalar batch term)."""
    if mu1 is None or logvar1 is None:
        kld = -0.5 * torch.sum(1 - logvar0.exp() - mu0.pow(2) + logvar0)
    else:
        kld = -0.5 * torch.sum(
            1 - logvar0.exp() / logvar1.exp() - (mu0 - mu1).pow(2) / logvar1.exp() + logvar0 - logvar1
        )
    return torch.mean(kld)


def poe(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    precision = 1.0 / var
    pd_mu = torch.sum(mu * precision, dim=0) / torch.sum(precision, dim=0)
    pd_var = 1.0 / torch.sum(precision, dim=0)
    pd_logvar = torch.log(pd_var + eps)
    return pd_mu, pd_logvar


def cal_loss_infonce(temperature, emb1, emb2, emb3):
    batch_size = emb1.shape[0]
    norm_emb1 = F.normalize(emb1, p=2, dim=-1)
    norm_emb2 = F.normalize(emb2, p=2, dim=-1)
    norm_emb3 = F.normalize(emb3, p=2, dim=-1)
    pos_sim = torch.sum(norm_emb1 * norm_emb2, dim=-1) / temperature
    neg_view_sim = torch.mm(norm_emb1, norm_emb3.t()) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_view_sim], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=emb1.device)
    return F.cross_entropy(logits, labels)


class M2VAE(BaseColdStartTrainer):
    def __init__(self, config):
        super(M2VAE, self).__init__(config)
        if self.args.cold_object == 'user':
            raise Exception('Cold user is not supported in M2VAE due to its item-focused multimodal design.')
        self.model = M2VAE_Learner(self.args, self.data, self.device)
        # Pretrain path may override args.emb_size / implicit_dim to match checkpoint.
        self.emb_size = self.args.emb_size

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.args.m2vae_weight_decay,
        )
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            batch_iter = next_batch_pairwise_CCFCRec(
                self.data,
                self.batch_size,
                self.args.positive_number,
                self.args.negative_number,
                self.args.self_neg_number
            )
            for n, batch in enumerate(batch_iter):
                u_idx, i_idx, neg_u_idx, pos_i_list, neg_i_list, _ = batch
                u_idx = torch.tensor(u_idx, dtype=torch.long, device=self.device)
                i_idx = torch.tensor(i_idx, dtype=torch.long, device=self.device)
                neg_u_idx = torch.tensor(neg_u_idx, dtype=torch.long, device=self.device)
                pos_i_list = torch.tensor(pos_i_list, dtype=torch.long, device=self.device)
                neg_i_list = torch.tensor(neg_i_list, dtype=torch.long, device=self.device)

                batch_loss = model.loss(
                    u_idx=u_idx,
                    i_idx=i_idx,
                    neg_u_idx=neg_u_idx,
                    pos_i_list=pos_i_list,
                    neg_i_list=neg_i_list,
                )
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                self.user_emb = model.user_embedding.clone()
                self.item_emb = model.item_embedding.clone()
                cold_item_gen = model.generate_item_emb(self.data.mapped_cold_item_idx)
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_item_gen
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag and self.early_stop_patience <= 0:
                        break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(self.user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def save(self):
        with torch.no_grad():
            self.best_user_emb = self.model.user_embedding.clone()
            self.best_item_emb = self.model.item_embedding.clone()
            cold_item_gen = self.model.generate_item_emb(self.data.mapped_cold_item_idx)
            self.best_item_emb.data[self.data.mapped_cold_item_idx] = cold_item_gen

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


class M2VAE_Learner(nn.Module):
    def __init__(self, args, data, device):
        super(M2VAE_Learner, self).__init__()
        self.args = args
        self.data = data
        self.device = device

        self._pretrain_ui_cpu = None
        if args.m2vae_pretrain:
            u_path = (
                f'./emb/{args.dataset}_cold_{args.cold_object}_{args.backbone}_user_emb.pt'
            )
            i_path = (
                f'./emb/{args.dataset}_cold_{args.cold_object}_{args.backbone}_item_emb.pt'
            )
            u0 = torch.load(u_path, map_location='cpu')
            i0 = torch.load(i_path, map_location='cpu')
            du, di = u0.shape[1], i0.shape[1]
            if du != di:
                raise ValueError(
                    f'M2VAE --m2vae_pretrain: user embedding dim {du} != item embedding dim {di}'
                )
            d_pre = du
            if (
                d_pre != args.implicit_dim
                or d_pre != args.attr_present_dim
                or d_pre != args.cat_implicit_dim
            ):
                warnings.warn(
                    f'M2VAE: pretrained embedding dim is {d_pre}; overriding implicit_dim, '
                    f'attr_present_dim, cat_implicit_dim to match (CLI had {args.implicit_dim}, '
                    f'{args.attr_present_dim}, {args.cat_implicit_dim}).',
                    UserWarning,
                    stacklevel=2,
                )
            setattr(args, 'implicit_dim', d_pre)
            setattr(args, 'attr_present_dim', d_pre)
            setattr(args, 'cat_implicit_dim', d_pre)
            setattr(args, 'emb_size', d_pre)
            self._pretrain_ui_cpu = (u0, i0)

        self.implicit_dim = args.implicit_dim
        self.attr_present_dim = args.attr_present_dim
        self.cat_implicit_dim = args.cat_implicit_dim

        self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        self.content_dim = self.item_content.shape[1]

        self.attr_matrix = nn.Parameter(torch.empty(self.content_dim, self.attr_present_dim))
        self.attr_W1 = nn.Parameter(torch.empty(self.attr_present_dim, self.attr_present_dim))
        self.attr_b1 = nn.Parameter(torch.empty(self.attr_present_dim, 1))
        self.attr_W2 = nn.Parameter(torch.empty(self.attr_present_dim, 1))
        self.h = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.content_projection = nn.Linear(self.content_dim, self.implicit_dim, bias=False)

        self.moe_gate_a = nn.Sequential(
            nn.Linear(self.implicit_dim, self.implicit_dim // 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.implicit_dim // 2, 1, bias=True),
        )
        self.moe_gate_c = nn.Sequential(
            nn.Linear(self.implicit_dim, self.implicit_dim // 2, bias=True),
            nn.Tanh(),
            nn.Linear(self.implicit_dim // 2, 1, bias=True),
        )
        self.user_gate_weights = nn.Parameter(torch.empty(2, self.implicit_dim, self.implicit_dim))
        self.user_gate_bias = nn.Parameter(torch.empty(2, self.implicit_dim))
        self.view_gate = nn.Sequential(
            nn.Linear(self.implicit_dim * 2, self.implicit_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(self.implicit_dim, 1, bias=True),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.attr_present_dim * 2, self.cat_implicit_dim),
            nn.BatchNorm1d(num_features=self.attr_present_dim),
            nn.Tanh(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.attr_present_dim + self.implicit_dim + self.implicit_dim, self.cat_implicit_dim),
            nn.BatchNorm1d(num_features=self.cat_implicit_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.cat_implicit_dim, self.implicit_dim),
            nn.BatchNorm1d(num_features=self.implicit_dim),
            nn.Tanh(),
        )

        self.mean_encoder_a_q = nn.Linear(self.attr_present_dim, self.implicit_dim)
        self.log_v_encoder_a_q = nn.Linear(self.attr_present_dim, self.implicit_dim)
        self.mean_encoder_c_q = nn.Linear(self.implicit_dim, self.implicit_dim)
        self.log_v_encoder_c_q = nn.Linear(self.implicit_dim, self.implicit_dim)
        self.mean_encoder_i = nn.Linear(self.implicit_dim, self.implicit_dim)
        self.log_v_encoder_i = nn.Linear(self.implicit_dim, self.implicit_dim)
        self.mean_encoder_a_p = nn.Linear(self.implicit_dim, self.implicit_dim)
        self.log_v_encoder_a_p = nn.Linear(self.implicit_dim, self.implicit_dim)

        self.user_embedding, self.item_embedding = self._init_id_embeddings()
        self._init_param()

    def _init_id_embeddings(self):
        if self.args.m2vae_pretrain:
            user_init, item_init = self._pretrain_ui_cpu
            user_emb = nn.Parameter(user_init, requires_grad=self.args.m2vae_pretrain_update)
            item_emb = nn.Parameter(item_init, requires_grad=self.args.m2vae_pretrain_update)
        else:
            user_emb = nn.Parameter(torch.empty(self.data.user_num, self.implicit_dim))
            item_emb = nn.Parameter(torch.empty(self.data.item_num, self.implicit_dim))
            nn.init.xavier_normal_(user_emb)
            nn.init.xavier_normal_(item_emb)
        return user_emb, item_emb

    def _init_param(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        nn.init.xavier_normal_(self.content_projection.weight)
        nn.init.xavier_uniform_(self.user_gate_weights)
        nn.init.zeros_(self.user_gate_bias)
        for module in [self.moe_gate_a, self.moe_gate_c]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        for layer in self.fc2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _encode_item_views(self, i_idx):
        attribute = self.item_content[i_idx]
        batch_size = i_idx.shape[0]
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1) + self.attr_b1.squeeze(), self.attr_W2).squeeze(-1)
        z_v = z_v.unsqueeze(0).repeat(batch_size, 1)
        if getattr(self.args, 'm2vae_attr_mask_neg1', False):
            neg_inf = torch.full(z_v.shape, -1e6, device=z_v.device, dtype=z_v.dtype)
            z_v = torch.where(attribute != -1, z_v, neg_inf)
        else:
            valid_mask = attribute != 0
            all_zero_rows = torch.sum(valid_mask, dim=1) == 0
            if torch.any(all_zero_rows):
                valid_mask[all_zero_rows] = True
            z_v = z_v.masked_fill(~valid_mask, -1e6)
        attr_attention_weight = torch.softmax(z_v, dim=1)
        a_v = torch.matmul(attr_attention_weight, self.attr_matrix)
        c_v = self.content_projection(attribute)
        return a_v, c_v

    def _decode_training_item(self, user_emb, i_idx):
        a_v, c_v = self._encode_item_views(i_idx)
        i_v = self.item_embedding[i_idx]

        mean_i = self.mean_encoder_i(i_v)
        log_i = self.log_v_encoder_i(i_v)
        mean_a_q = self.mean_encoder_a_q(a_v)
        log_a_q = self.log_v_encoder_a_q(a_v)
        z_a = mean_a_q + torch.exp(log_a_q * 0.5) * torch.randn_like(mean_a_q)

        h1 = self.fc2(torch.cat([a_v, c_v], dim=-1))
        mean_p = self.mean_encoder_a_p(h1)
        log_p = self.log_v_encoder_a_p(h1)
        mean_c_q = self.mean_encoder_c_q(c_v)
        log_c_q = self.log_v_encoder_c_q(c_v)
        z_c = mean_c_q + torch.exp(log_c_q * 0.5) * torch.randn_like(mean_c_q)

        mean_p1, log_p1 = poe(torch.stack([mean_a_q, mean_c_q], dim=0), torch.stack([log_a_q, log_c_q], dim=0))
        z_p = mean_p1 + torch.exp(log_p1 * 0.5) * torch.randn_like(mean_p1)

        user_exp = user_emb.unsqueeze(0).expand(2, -1, -1)
        user_gate_logit = torch.einsum('mbd,mdh->mbh', user_exp, self.user_gate_weights) + self.user_gate_bias.unsqueeze(1)
        user_gates = torch.sigmoid(user_gate_logit)
        gated_users = user_emb.unsqueeze(0) * user_gates
        user_emb_a, user_emb_c = gated_users[0], gated_users[1]

        moe_gate_a = self.moe_gate_a(user_emb_a * z_a * z_p)
        moe_gate_c = self.moe_gate_c(user_emb_c * z_c * z_p)
        moe_weights = torch.softmax(torch.cat([moe_gate_a, moe_gate_c], dim=1), dim=1)
        moe_a = moe_weights[:, 0:1]
        moe_c = moe_weights[:, 1:2]
        mean_u = moe_a * mean_a_q + moe_c * mean_c_q
        log_u = moe_a * log_a_q + moe_c * log_c_q

        com_weights = self.view_gate(torch.cat([user_emb, z_p], dim=1))
        mean_q = com_weights * mean_p1 + (1 - com_weights) * mean_u
        log_q = com_weights * log_p1 + (1 - com_weights) * log_u

        mean_mopoe = 0.5 * (mean_q + mean_i)
        log_mopoe = 0.5 * (log_q + log_i)
        z = mean_mopoe + torch.exp(log_mopoe * 0.5) * torch.randn_like(mean_mopoe)
        decoder = self.decoder(self.fc3(torch.cat([z, a_v, c_v], dim=-1)))
        return {
            'decoder': decoder,
            'a_v': a_v,
            'c_v': c_v,
            'i_v': i_v,
            'z_a': z_a,
            'z_c': z_c,
            'z_p': z_p,
            'mean_i': mean_i,
            'log_i': log_i,
            'mean_p': mean_p,
            'log_p': log_p,
            'mean_a_q': mean_a_q,
            'log_a_q': log_a_q,
            'mean_c_q': mean_c_q,
            'log_c_q': log_c_q,
            'mean_mopoe': mean_mopoe,
            'log_mopoe': log_mopoe,
        }

    def _contrast_sum_term(self, decoder, pos_item_emb, neg_item_emb):
        """sum_b sum_p L_bp / P (model_hi_moe train)."""
        decoder_u = decoder.unsqueeze(1)
        p = pos_item_emb.shape[1]
        pos_mul = torch.sum(decoder_u * pos_item_emb, dim=2) / (
            self.args.tau * torch.norm(decoder_u, dim=2) * torch.norm(pos_item_emb, dim=2)
        )
        pos_exp = torch.exp(pos_mul)
        du2 = decoder_u.unsqueeze(1)
        neg_mul = torch.sum(du2 * neg_item_emb, dim=3) / (
            self.args.tau * torch.norm(du2, dim=3) * torch.norm(neg_item_emb, dim=3)
        )
        neg_sum = torch.sum(torch.exp(neg_mul), dim=2)
        contrast_val = -torch.log(pos_exp / (pos_exp + neg_sum))
        return torch.sum(torch.sum(contrast_val, dim=1), dim=0) / p

    def _rank_sum_term(self, decoder, pos_user, neg_user):
        logsigmoid = nn.LogSigmoid()
        y_uv = torch.sum(decoder * pos_user, dim=1)
        y_kv = torch.sum(decoder * neg_user, dim=1)
        return -logsigmoid(y_uv - y_kv).sum()

    def loss(self, u_idx, i_idx, neg_u_idx, pos_i_list, neg_i_list):
        out = self._decode_training_item(self.user_embedding[u_idx], i_idx)
        decoder = out['decoder']

        pos_item_emb = self.item_embedding[pos_i_list]
        neg_item_emb = self.item_embedding[neg_i_list]
        pos_user_emb = self.user_embedding[u_idx]
        neg_user_emb = self.user_embedding[neg_u_idx]

        decouple_loss = cal_loss_infonce(self.args.tau, out['z_a'], out['a_v'], out['z_p']) + cal_loss_infonce(
            self.args.tau, out['z_c'], out['c_v'], out['z_p']
        )

        kld = calc_kl_divergence(out['mean_mopoe'], out['log_mopoe'], out['mean_p'], out['log_p'])
        kld = kld + (
            calc_kl_divergence(out['mean_i'], out['log_i'])
            + calc_kl_divergence(out['mean_c_q'], out['log_c_q'])
            + calc_kl_divergence(out['mean_a_q'], out['log_a_q'])
        ) / 3.0
        recon = torch.mean(torch.sum((out['i_v'] - decoder) ** 2, dim=1))
        contrast_sum = self._contrast_sum_term(decoder, pos_item_emb, neg_item_emb)
        rank_sum = self._rank_sum_term(decoder, pos_user_emb, neg_user_emb)

        return (
            self.args.m2vae_kld_weight * kld
            + self.args.m2vae_recon_weight * recon
            + rank_sum
            + contrast_sum
            + self.args.m2vae_decouple_weight * decouple_loss
        )

    def generate_item_emb(self, idx):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        a_v, c_v = self._encode_item_views(idx)
        i_v = self.item_embedding[idx]
        mean_i = self.mean_encoder_i(i_v)
        log_i = self.log_v_encoder_i(i_v)
        mean_a_q = self.mean_encoder_a_q(a_v)
        log_a_q = self.log_v_encoder_a_q(a_v)
        mean_c_q = self.mean_encoder_c_q(c_v)
        log_c_q = self.log_v_encoder_c_q(c_v)
        mean_p1, log_p1 = poe(torch.stack([mean_a_q, mean_c_q], dim=0), torch.stack([log_a_q, log_c_q], dim=0))
        mean_mopoe = 0.5 * (mean_p1 + mean_i)
        log_mopoe = 0.5 * (log_p1 + log_i)
        z = mean_mopoe + torch.exp(log_mopoe * 0.5) * torch.randn_like(mean_mopoe)
        return self.decoder(self.fc3(torch.cat([z, a_v, c_v], dim=-1)))

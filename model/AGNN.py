"""
AGNN for ColdRec: single-vector content, precomputed warm-pool cosine KNN,
official-style add/erase gates + LeakyReLU neighbor fusion, dual 2-layer VAE (Q/P),
implicit BPR + gen-BPR + align + ELBO (ColdRec protocol).

User vs item content/KNN are separate (item cold-start still runs a user branch; no
longer indexes user ids into the item content matrix).

Other baselines are unaffected (AGNN-only module).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss, mse_loss


class VAE2L(nn.Module):
    """Two-layer encoder/decoder VAE (same layout as official AGNN VAE: X->h->mu,logvar; z->h->X)."""

    def __init__(self, embed_size):
        super().__init__()
        d = embed_size
        self.enc1 = nn.Linear(d, d)
        self.enc_mu = nn.Linear(d, d)
        self.enc_logvar = nn.Linear(d, d)
        self.dec1 = nn.Linear(d, d)
        self.dec2 = nn.Linear(d, d)

    def Q(self, x):
        h = F.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparam(self, mu, logvar, training):
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def P(self, z):
        h = F.relu(self.dec1(z))
        return self.dec2(h)


def _cosine_knn_indices(feature_np, warm_ids_np, k, exclude_self=True):
    """
    For each global entity id 0..n-1, KNN over warm pool by cosine on feature_np rows.
    feature_np: (n, F)
    Returns [n, k] global indices.
    """
    n = feature_np.shape[0]
    warm_ids_np = np.asarray(warm_ids_np, dtype=np.int64)
    nw = warm_ids_np.shape[0]
    if nw == 0:
        raise ValueError('AGNN: warm pool is empty for KNN.')
    kk = min(int(k), nw)
    Wmat = feature_np[warm_ids_np]
    Wn = Wmat / (np.linalg.norm(Wmat, axis=1, keepdims=True) + 1e-12)
    knn_idx = np.zeros((n, k), dtype=np.int64)
    warm_set = set(warm_ids_np.tolist())
    for i in range(n):
        q = feature_np[i : i + 1]
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        sim = (qn @ Wn.T).ravel()
        if exclude_self and i in warm_set:
            pos = np.where(warm_ids_np == i)[0]
            if pos.size > 0:
                sim[pos[0]] = -np.inf
        if kk == nw:
            order = np.argsort(-sim)[:kk]
        else:
            part = np.argpartition(-sim, kk - 1)[:kk]
            order = part[np.argsort(-sim[part])]
        chosen = warm_ids_np[order]
        if chosen.size < k:
            pad = np.full(k - chosen.size, chosen[-1], dtype=np.int64)
            chosen = np.concatenate([chosen, pad])
        knn_idx[i] = chosen[:k]
    return knn_idx


def _pad_side_info(mapped_np, num_rows):
    """Pad or truncate mapped side info to num_rows (align with id embedding rows)."""
    if mapped_np is None:
        return None
    mapped_np = np.asarray(mapped_np, dtype=np.float32)
    f = mapped_np.shape[1]
    out = np.zeros((num_rows, f), dtype=np.float32)
    n = min(num_rows, mapped_np.shape[0])
    out[:n] = mapped_np[:n]
    return out


class AGNN_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super().__init__()
        self.args = args
        self.data = data
        self.latent_size = emb_size
        self.device = device
        n_u, n_i = data.user_num, data.item_num

        self.embedding_dict = self._init_id_embeddings()

        item_np = _pad_side_info(data.mapped_item_content, n_i)
        user_np = _pad_side_info(data.mapped_user_content, n_u)
        self._has_item_content = item_np is not None
        self._has_user_content = user_np is not None
        if not self._has_item_content and args.cold_object == 'item':
            raise ValueError('AGNN item cold-start requires item content features.')
        if not self._has_user_content and args.cold_object == 'user':
            raise ValueError('AGNN user cold-start requires user content features.')

        self._item_content_cpu = torch.from_numpy(item_np).float().contiguous() if item_np is not None else None
        self._user_content_cpu = torch.from_numpy(user_np).float().contiguous() if user_np is not None else None

        warm_item = np.asarray(data.mapped_warm_item_idx, dtype=np.int64)
        warm_user = np.asarray(data.mapped_warm_user_idx, dtype=np.int64)
        knn_k_req = min(args.agnn_knn_k, max(1, warm_item.shape[0]))
        knn_k_user = min(args.agnn_knn_k, max(1, warm_user.shape[0]))

        # Features for KNN: content rows, or initial id embeddings if content missing for that entity type.
        item_feat_np = item_np if item_np is not None else self.embedding_dict['item_emb'].detach().cpu().numpy()
        user_feat_np = user_np if user_np is not None else self.embedding_dict['user_emb'].detach().cpu().numpy()

        knn_item = _cosine_knn_indices(item_feat_np, warm_item, knn_k_req)
        knn_user = _cosine_knn_indices(user_feat_np, warm_user, knn_k_user)
        self._knn_item_cpu = torch.from_numpy(knn_item).long().contiguous()
        self._knn_user_cpu = torch.from_numpy(knn_user).long().contiguous()

        drop = args.agnn_dropout
        if item_np is not None:
            f_i = item_np.shape[1]
            self.encoder_item = nn.Sequential(
                nn.Linear(f_i, 2 * self.latent_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(2 * self.latent_size, self.latent_size),
            )
        else:
            self.encoder_item = self._id_mlp_stub(drop)
        if user_np is not None:
            f_u = user_np.shape[1]
            self.encoder_user = nn.Sequential(
                nn.Linear(f_u, 2 * self.latent_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(2 * self.latent_size, self.latent_size),
            )
        else:
            self.encoder_user = self._id_mlp_stub(drop)

        self.dense_self = nn.Linear(2 * self.latent_size, self.latent_size)
        self.dense_hop = nn.Linear(2 * self.latent_size, self.latent_size)
        self.dense_addgate = nn.Linear(2 * self.latent_size, self.latent_size)
        self.dense_erasegate = nn.Linear(2 * self.latent_size, self.latent_size)
        self.leaky = nn.LeakyReLU(0.2)

        self.user_vae = VAE2L(self.latent_size)
        self.item_vae = VAE2L(self.latent_size)

    def _id_mlp_stub(self, drop):
        """When no side info for an entity type: map id embedding -> latent semantic."""
        return nn.Sequential(
            nn.Linear(self.latent_size, 2 * self.latent_size),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(2 * self.latent_size, self.latent_size),
        )

    def _init_id_embeddings(self):
        if getattr(self.args, 'agnn_no_backbone', False):
            ud = nn.Parameter(torch.empty(self.data.user_num, self.latent_size))
            vd = nn.Parameter(torch.empty(self.data.item_num, self.latent_size))
            nn.init.xavier_uniform_(ud)
            nn.init.xavier_uniform_(vd)
            return nn.ParameterDict({'user_emb': ud, 'item_emb': vd})
        user_weight = torch.load(
            f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt',
            map_location='cpu',
        )
        item_weight = torch.load(
            f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt',
            map_location='cpu',
        )
        return nn.ParameterDict({
            'user_emb': nn.Parameter(user_weight, requires_grad=self.args.agnn_finetune),
            'item_emb': nn.Parameter(item_weight, requires_grad=self.args.agnn_finetune),
        })

    def _gather_item_content(self, idx):
        dev = idx.device
        ic = idx.detach().cpu()
        return self._item_content_cpu[ic].to(dev, non_blocking=False)

    def _gather_user_content(self, idx):
        dev = idx.device
        ic = idx.detach().cpu()
        return self._user_content_cpu[ic].to(dev, non_blocking=False)

    def _gather_knn(self, idx, is_user):
        dev = idx.device
        ic = idx.detach().cpu()
        t = self._knn_user_cpu if is_user else self._knn_item_cpu
        return t[ic].to(dev, non_blocking=False)

    def _semantic_x(self, idx, is_user):
        if is_user:
            if self._has_user_content:
                return self.encoder_user(self._gather_user_content(idx))
            return self.encoder_user(self.embedding_dict['user_emb'][idx])
        if self._has_item_content:
            return self.encoder_item(self._gather_item_content(idx))
        return self.encoder_item(self.embedding_dict['item_emb'][idx])

    def _entity_gcn_embed(self, idx, is_user):
        x = self._semantic_x(idx, is_user)
        if is_user:
            e = self.embedding_dict['user_emb'][idx]
        else:
            e = self.embedding_dict['item_emb'][idx]

        self_embed = self.dense_self(torch.cat([x, e], dim=1))
        neigh_glob = self._gather_knn(idx, is_user)
        if is_user:
            neigh_e = self.embedding_dict['user_emb'][neigh_glob]
        else:
            neigh_e = self.embedding_dict['item_emb'][neigh_glob]

        B, K, D = neigh_e.shape
        self_rep = self_embed.unsqueeze(1).expand(B, K, D)
        hop_raw = self.dense_hop(torch.cat([self_rep, neigh_e], dim=-1))

        add_gate = torch.sigmoid(self.dense_addgate(torch.cat([self_rep, hop_raw], dim=-1)))
        hop_g = (hop_raw * add_gate).mean(dim=1)

        erase_gate = torch.sigmoid(self.dense_erasegate(torch.cat([self_embed, hop_raw.mean(dim=1)], dim=1)))
        self_erased = (1.0 - erase_gate) * self_embed

        return self.leaky(self_erased + hop_g)

    def forward_gcn_user(self, uid):
        return self._entity_gcn_embed(uid.long(), is_user=True)

    def forward_gcn_item(self, iid):
        return self._entity_gcn_embed(iid.long(), is_user=False)

    def _vae_loss_pair(self, self_feature, id_target, vae, training):
        mu, logvar = vae.Q(self_feature)
        z = vae.reparam(mu, logvar, training)
        recon = vae.P(z)
        recon_loss = mse_loss(recon, id_target)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return recon, recon_loss, kl_loss

    def _vae_input_item(self, idx):
        return self._semantic_x(idx, is_user=False)

    def _vae_input_user(self, idx):
        return self._semantic_x(idx, is_user=True)

    def generate_item_emb(self, idx, training=True):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device)
        x = self._vae_input_item(idx)
        mu, logvar = self.item_vae.Q(x)
        z = self.item_vae.reparam(mu, logvar, training)
        return self.item_vae.P(z)

    def generate_user_emb(self, idx, training=True):
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.device)
        else:
            idx = idx.to(self.device)
        x = self._vae_input_user(idx)
        mu, logvar = self.user_vae.Q(x)
        z = self.user_vae.reparam(mu, logvar, training)
        return self.user_vae.P(z)

    def loss(self, uid, iid, nid, reg):
        uid = uid.long()
        iid = iid.long()
        nid = nid.long()

        u_g = self.forward_gcn_user(uid)
        pos_g = self.forward_gcn_item(iid)
        neg_g = self.forward_gcn_item(nid)

        rank_loss = bpr_loss(u_g, pos_g, neg_g)

        if self.args.cold_object == 'item':
            x_pos = self._vae_input_item(iid)
            target = self.embedding_dict['item_emb'][iid]
            gen_pos, recon_loss, kl_loss = self._vae_loss_pair(x_pos, target, self.item_vae, True)
            gen_rank_loss = bpr_loss(u_g, gen_pos, neg_g)
            align_loss = mse_loss(gen_pos, target)
            reg_loss = l2_reg_loss(reg, u_g, pos_g, neg_g, gen_pos)
        else:
            x_u = self._vae_input_user(uid)
            target = self.embedding_dict['user_emb'][uid]
            gen_u, recon_loss, kl_loss = self._vae_loss_pair(x_u, target, self.user_vae, True)
            gen_rank_loss = bpr_loss(gen_u, pos_g, neg_g)
            align_loss = mse_loss(gen_u, target)
            reg_loss = l2_reg_loss(reg, u_g, pos_g, neg_g, gen_u)

        return (
            rank_loss
            + self.args.agnn_rank_weight * gen_rank_loss
            + self.args.agnn_align_weight * align_loss
            + self.args.agnn_vae_lambda * (recon_loss + kl_loss)
            + reg_loss
        )

    def forward(self):
        device = self.embedding_dict['user_emb'].device
        n_u, n_i = self.data.user_num, self.data.item_num
        bs = max(1, int(getattr(self.args, 'agnn_eval_chunk', 512)))
        u_all = torch.empty(n_u, self.latent_size, device=device, dtype=self.embedding_dict['user_emb'].dtype)
        for s in range(0, n_u, bs):
            e = min(s + bs, n_u)
            u_all[s:e] = self.forward_gcn_user(torch.arange(s, e, device=device, dtype=torch.long))
        i_all = torch.empty(n_i, self.latent_size, device=device, dtype=self.embedding_dict['item_emb'].dtype)
        for s in range(0, n_i, bs):
            e = min(s + bs, n_i)
            i_all[s:e] = self.forward_gcn_item(torch.arange(s, e, device=device, dtype=torch.long))
        return u_all, i_all


class AGNN(BaseColdStartTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.model = AGNN_Learner(self.args, self.data, self.emb_size, self.device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
        )
        cold_item_idx = torch.tensor(self.data.mapped_cold_item_idx, dtype=torch.long, device=self.device)
        cold_user_idx = torch.tensor(self.data.mapped_cold_user_idx, dtype=torch.long, device=self.device)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.tensor(user_idx, dtype=torch.long, device=self.device)
                pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                neg_idx = torch.tensor(neg_idx, dtype=torch.long, device=self.device)
                batch_loss = model.loss(user_idx, pos_idx, neg_idx, self.reg)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = model.forward()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                ch = max(1, int(getattr(self.args, 'agnn_eval_chunk', 512)))
                if self.args.cold_object == 'item':
                    for s in range(0, cold_item_idx.numel(), ch):
                        e = min(s + ch, cold_item_idx.numel())
                        sl = cold_item_idx[s:e]
                        self.item_emb[sl] = model.generate_item_emb(sl, training=False)
                else:
                    for s in range(0, cold_user_idx.numel(), ch):
                        e = min(s + ch, cold_user_idx.numel())
                        sl = cold_user_idx[s:e]
                        self.user_emb[sl] = model.generate_user_emb(sl, training=False)
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
            u, v = self.model.forward()
            self.best_user_emb = u.clone()
            self.best_item_emb = v.clone()
            cold_item_idx = torch.tensor(self.data.mapped_cold_item_idx, dtype=torch.long, device=self.device)
            cold_user_idx = torch.tensor(self.data.mapped_cold_user_idx, dtype=torch.long, device=self.device)
            ch = max(1, int(getattr(self.args, 'agnn_eval_chunk', 512)))
            if self.args.cold_object == 'item':
                for s in range(0, cold_item_idx.numel(), ch):
                    e = min(s + ch, cold_item_idx.numel())
                    sl = cold_item_idx[s:e]
                    self.best_item_emb[sl] = self.model.generate_item_emb(sl, training=False)
            else:
                for s in range(0, cold_user_idx.numel(), ch):
                    e = min(s + ch, cold_user_idx.numel())
                    sl = cold_user_idx[s:e]
                    self.best_user_emb[sl] = self.model.generate_user_emb(sl, training=False)

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def batch_predict(self, users):
        with torch.no_grad():
            users = self.data.get_user_id_list(users)
            users = torch.tensor(users, device=self.device)
            return torch.matmul(self.user_emb[users], self.item_emb.transpose(0, 1))

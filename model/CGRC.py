"""
CGRC: Content-based Graph Reconstruction for Cold-start Item Recommendation (SIGIR 2024).
Single-vector item content (no multimodal L_M). Training: L_E on masked G', L_R on full G (teacher forcing).
Inference: top-K edge reconstruction then LightGCN on Ĝ (paper Sec. 3.4.1).
"""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.databuilder import TorchGraphInterface
from util.utils import l2_reg_loss, next_batch_cgrc


def _bip_adj_from_R(R, n_users, n_items):
    """Symmetric bipartite adjacency from user×item interaction matrix R (CSR)."""
    R = R.tocoo()
    rows, cols = R.row, R.col
    data = np.ones(len(rows), dtype=np.float32)
    row_idx = np.concatenate([rows, cols + n_users])
    col_idx = np.concatenate([cols + n_users, rows])
    data_sym = np.concatenate([data, data])
    return sp.csr_matrix((data_sym, (row_idx, col_idx)), shape=(n_users + n_items, n_users + n_items))


def _drop_edges_to_items(R_csr, cold_items):
    """Remove all edges incident to item columns in cold_items (Eq. 3)."""
    coo = R_csr.tocoo()
    cold = np.asarray(list(cold_items), dtype=np.int64)
    mask = ~np.isin(coo.col, cold)
    return sp.csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=R_csr.shape)


def _add_edges_to_R(R_csr, pairs):
    """Union of R_csr with unit-weight edges pairs (mapped user, item).

    Equivalent to repeatedly assigning lil[u, i] = 1.0: duplicate (u, i) in pairs
    or already present in R stay binary 1 after coalescing.
    """
    if not pairs:
        return R_csr.tocsr(copy=True)
    coo = R_csr.tocoo()
    n_new = len(pairs)
    pu = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=n_new)
    pi = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=n_new)
    row = np.concatenate([coo.row.astype(np.int64, copy=False), pu])
    col = np.concatenate([coo.col.astype(np.int64, copy=False), pi])
    data = np.concatenate(
        [np.asarray(coo.data, dtype=np.float32), np.ones(n_new, dtype=np.float32)]
    )
    out = sp.csr_matrix((data, (row, col)), shape=R_csr.shape, dtype=np.float32)
    out.eliminate_zeros()
    if out.nnz:
        out.data = np.minimum(out.data, 1.0)
    return out


def _sparse_adj_tensor(adj_mat, device):
    return TorchGraphInterface.convert_sparse_mat_to_tensor(adj_mat).to(device)


def _lightgcn_mean_all_layers(adj_t, user_emb, item_x, n_users, n_layers):
    """LightGCN on full graph: mean of embeddings from layers 0..L (inclusive)."""
    ego = torch.cat([user_emb, item_x], dim=0)
    embs = [ego]
    h = ego
    for _ in range(n_layers):
        h = torch.sparse.mm(adj_t, h)
        embs.append(h)
    out = torch.stack(embs, dim=1).mean(dim=1)
    return out[:n_users], out[n_users:]


def _propagate_gprime_frozen_cold(adj_t, user_emb, item_x, n_users, n_layers, cold_item_idx):
    """
    LightGCN on G'; cold items stay at content x_i each layer after conv (Sec. 3.3.2).
    Returns list h^{(0)}..h^{(L)} each (n_users+n_items, d).
    """
    ego = torch.cat([user_emb, item_x], dim=0)
    out = [ego]
    h = ego
    if cold_item_idx.numel() > 0:
        cold_rows = cold_item_idx + n_users
    else:
        cold_rows = None
    for _ in range(n_layers):
        h = torch.sparse.mm(adj_t, h)
        if cold_rows is not None:
            h[cold_rows] = item_x[cold_item_idx]
        out.append(h)
    return out


def _user_mean_layers_1_to_L(layer_list, n_users, L):
    """Eq. (5): mean of user rows for layers 1..L (exclude layer 0).

    Each layer_list[l] is (n_users + n_items, d); only the first n_users rows are
    user embeddings. Averaging the full tensor would mix item nodes and break
    edge_logits_broadcast / top-k user selection for Ĝ.
    """
    if L <= 0:
        return layer_list[0][:n_users]
    stacks = torch.stack([layer_list[l][:n_users] for l in range(1, L + 1)], dim=0)
    return stacks.mean(dim=0)


class CGRCNet(nn.Module):
    def __init__(self, n_users, n_items, content_dim, emb_dim, mlp_hidden, item_content_np):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.register_buffer('item_content', torch.from_numpy(item_content_np).float())
        self.user_emb = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_users, emb_dim)))
        self.item_lin = nn.Linear(content_dim, emb_dim, bias=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def item_x(self, item_content=None):
        c = self.item_content if item_content is None else item_content
        return self.item_lin(c)

    def edge_logits_broadcast(self, h_u_bar, x_cold, cold_ids):
        """
        h_u_bar: (U', d) — any subset of user rows (training: users with masked edges only).
        x_cold: full item_x (I, d), cold_ids: (C,) item indices.
        returns S: (U', C)
        """
        xc = x_cold[cold_ids]
        U, d = h_u_bar.shape
        C = xc.shape[0]
        hu = h_u_bar.unsqueeze(1).expand(U, C, d).reshape(U * C, d)
        xi = xc.unsqueeze(0).expand(U, C, d).reshape(U * C, d)
        logits = self.edge_mlp(torch.cat([hu, xi], dim=1)).view(U, C)
        return logits


class CGRC(BaseColdStartTrainer):
    def __init__(self, config):
        super().__init__(config)
        if self.args.cold_object != 'item':
            raise ValueError('CGRC is defined for item cold-start only (--cold_object item).')
        if self.data.mapped_item_content is None:
            raise ValueError('CGRC requires item content (dataset item_content.npy).')
        self.Lg = self.args.cgrc_layers_gprime
        self.Lf = self.args.cgrc_layers_full
        self.Lh = self.args.cgrc_layers_ghat
        self.rho = self.args.cgrc_mask_rho
        self.topk = self.args.cgrc_recon_topk
        self.lambda_e = self.args.cgrc_lambda_e
        self.tau = self.args.cgrc_tau
        self.le_max = self.args.cgrc_le_max_edges
        self.ranking_neg = self.args.cgrc_ranking_neg_per_user

        cdim = int(self.data.mapped_item_content.shape[1])
        self.model = CGRCNet(
            self.data.user_num,
            self.data.item_num,
            cdim,
            self.emb_size,
            self.args.cgrc_mlp_hidden,
            self.data.mapped_item_content,
        )
        self.sparse_full = _sparse_adj_tensor(self.data.norm_adj, self.device)

        R = self.data.interaction_mat.tocsr()
        _coo = R.tocoo()
        self._R_coo_row = _coo.row.astype(np.int64)
        self._R_coo_col = _coo.col.astype(np.int64)

        self._eligible_items = torch.from_numpy(np.asarray(R.getnnz(axis=0) > 0).nonzero()[0]).long()
        self._user_rated = []
        for uid in range(self.data.user_num):
            u = self.data.id2user[uid]
            self._user_rated.append(frozenset(self.data.item[it] for it in self.data.training_set_u[u].keys()))

        self.best_state_dict = None

    def _sample_cold_items(self):
        """Bernoulli(rho) over items with >=1 training interaction."""
        el = self._eligible_items
        if el.numel() == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        mask = torch.rand(el.shape[0]) < self.rho
        chosen = el[mask]
        if chosen.numel() == 0 and self.rho > 0 and el.numel() > 0:
            chosen = el[torch.randint(el.numel(), (1,))]
        return chosen.to(self.device)

    def _masked_edges(self, cold_ids_cpu):
        """List of (u, i) mapped for edges removed (incident to cold items)."""
        cold_arr = cold_ids_cpu.detach().numpy().astype(np.int64, copy=False)
        if cold_arr.size == 0:
            return []
        rows = self._R_coo_row
        cols = self._R_coo_col
        m = np.isin(cols, cold_arr)
        if not m.any():
            return []
        return list(zip(rows[m].tolist(), cols[m].tolist()))

    def _reconstruction_loss(self, logits, cold_ids, masked_edges, u_indices):
        """
        Eq. (6): mean over masked edges; denominator is cold items j not in user u's training set N_u.
        logits: (R, C) — row r is global user u_indices[r]; columns follow cold_ids order.
        u_indices: (R,) long on same device as logits.
        """
        if not masked_edges:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        cold_list = cold_ids.detach().cpu().tolist()
        col_of = {cid: c for c, cid in enumerate(cold_list)}
        R, _ = logits.shape
        device = logits.device

        row_map = {int(u_indices[r].item()): r for r in range(R)}
        by_user = defaultdict(list)
        for u, i in masked_edges:
            by_user[u].append(i)

        valid_rows = []
        for r in range(R):
            uid = int(u_indices[r].item())
            rated = self._user_rated[uid]
            valid_rows.append([cid not in rated for cid in cold_list])
        valid_mask = torch.tensor(valid_rows, device=device, dtype=torch.bool)

        neg_inf = torch.finfo(logits.dtype).min
        le = torch.logsumexp(logits.masked_fill(~valid_mask, neg_inf), dim=1)

        row_inds = []
        col_inds = []
        for u, items in by_user.items():
            r = row_map.get(u)
            if r is None:
                continue
            if not valid_mask[r].any():
                continue
            for i in items:
                if i not in col_of:
                    continue
                row_inds.append(r)
                col_inds.append(col_of[i])

        if not row_inds:
            return torch.zeros((), device=device, dtype=logits.dtype)

        row_t = torch.tensor(row_inds, device=device, dtype=torch.long)
        col_t = torch.tensor(col_inds, device=device, dtype=torch.long)
        pos = logits[row_t, col_t]
        le_t = le[row_t]
        n = row_t.numel()
        return -(pos - le_t).sum() / n

    def _ranking_loss(self, z_u, z_i, u_idx, i_pos, B_list):
        """Eq. (8): contrastive over minibatch item set B (vectorized logsumexp per row)."""
        device = z_u.device
        B = len(u_idx)
        if B == 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        L = len(B_list)
        B_t = torch.tensor(B_list, dtype=torch.long, device=device)
        z_u_b = z_u[u_idx]
        z_items = z_i[B_t]
        sim = torch.matmul(z_u_b, z_items.transpose(0, 1)) / self.tau

        B_map = {j: idx for idx, j in enumerate(B_list)}
        pos_cols = [B_map.get(i_pos[b], -1) for b in range(B)]
        pos_cols_t = torch.tensor(pos_cols, dtype=torch.long, device=device)

        neg_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            rated = self._user_rated[u_idx[b]]
            neg_mask[b] = torch.tensor([j not in rated for j in B_list], device=device, dtype=torch.bool)

        neg_inf = torch.finfo(sim.dtype).min
        le_row = torch.logsumexp(sim.masked_fill(~neg_mask, neg_inf), dim=1)
        has_pos = pos_cols_t >= 0
        has_neg = neg_mask.any(dim=1)
        valid = has_pos & has_neg
        if not valid.any():
            return torch.zeros((), device=device, dtype=sim.dtype)

        ar = torch.arange(B, device=device, dtype=torch.long)
        pos_s = sim[ar[valid], pos_cols_t[valid]]
        return -(pos_s - le_row[valid]).sum() / valid.sum().to(dtype=sim.dtype)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        R_base = self.data.interaction_mat.tocsr()
        n_users, n_items = self.data.user_num, self.data.item_num
        self.timer(start=True)
        epoch = -1
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_cgrc(self.data, self.batch_size, self.ranking_neg)):
                u_idx, i_idx, B_list = batch
                u_idx_t = torch.tensor(u_idx, dtype=torch.long, device=self.device)
                i_idx_t = torch.tensor(i_idx, dtype=torch.long, device=self.device)
                x_all = model.item_x()

                cold_ids = self._sample_cold_items()
                loss_e = torch.zeros((), device=self.device, dtype=torch.float32)
                if cold_ids.numel() > 0:
                    cold_cpu = cold_ids.detach().cpu()
                    edges = self._masked_edges(cold_cpu)
                    if len(edges) > self.le_max:
                        sel = np.random.choice(len(edges), self.le_max, replace=False)
                        edges = [edges[s] for s in sel]
                    if edges:
                        Rm = _drop_edges_to_items(R_base, set(cold_cpu.tolist()))
                        adj_m = self.data.normalize_graph_mat(_bip_adj_from_R(Rm, n_users, n_items))
                        adj_t = _sparse_adj_tensor(adj_m, self.device)
                        layers = _propagate_gprime_frozen_cold(
                            adj_t, model.user_emb, x_all, n_users, self.Lg, cold_ids
                        )
                        h_u_bar = _user_mean_layers_1_to_L(layers, n_users, self.Lg)
                        u_need = sorted({u for u, _ in edges})
                        u_tensor = torch.tensor(u_need, dtype=torch.long, device=self.device)
                        h_sub = h_u_bar[u_tensor]
                        logits_sub = model.edge_logits_broadcast(h_sub, x_all, cold_ids)
                        loss_e = self._reconstruction_loss(logits_sub, cold_ids, edges, u_tensor)

                z_u, z_i = _lightgcn_mean_all_layers(self.sparse_full, model.user_emb, x_all, n_users, self.Lf)
                loss_r = self._ranking_loss(z_u, z_i, u_idx, i_idx, B_list)

                reg = l2_reg_loss(
                    self.reg,
                    model.user_emb[u_idx_t],
                    x_all[i_idx_t],
                )
                batch_loss = self.lambda_e * loss_e + loss_r + reg
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print(
                        'training:', epoch + 1, 'batch', n,
                        'loss:', batch_loss.item(), 'L_E:', loss_e.item(), 'L_R:', loss_r.item(),
                    )

            with torch.no_grad():
                model.eval()
                if epoch % self.eval_every == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag and self.early_stop_patience <= 0:
                        break

        self.epochs_ran = (epoch + 1) if self.maxEpoch > 0 else 0
        self.timer(start=False)
        model.eval()
        if self.best_state_dict is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})
        if self.args.save_emb:
            with torch.no_grad():
                x_all = model.item_x()
                z_u, z_i = _lightgcn_mean_all_layers(self.sparse_full, model.user_emb, x_all, n_users, self.Lf)
            torch.save(z_u, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(z_i, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def save(self):
        with torch.no_grad():
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _forward_scores_ghat(self, model):
        """Build Ĝ for real cold items, run LightGCN; return (n_users, n_items) scores z_u @ z_i.T."""
        n_users, n_items = self.data.user_num, self.data.item_num
        R_base = self.data.interaction_mat.tocsr()
        x_all = model.item_x()

        if self.Lg > 0:
            layers = _propagate_gprime_frozen_cold(
                self.sparse_full, model.user_emb, x_all, n_users, self.Lg, torch.tensor([], device=self.device)
            )
            h_u_bar = _user_mean_layers_1_to_L(layers, n_users, self.Lg)
        else:
            h_u_bar = model.user_emb

        cold_np = self.data.mapped_cold_item_idx
        if len(cold_np) == 0:
            z_u, z_i = _lightgcn_mean_all_layers(self.sparse_full, model.user_emb, x_all, n_users, self.Lh)
            return torch.matmul(z_u, z_i.transpose(0, 1))

        cold_t = torch.tensor(cold_np, dtype=torch.long, device=self.device)
        logits = model.edge_logits_broadcast(h_u_bar, x_all, cold_t)
        k = min(self.topk, n_users)
        _, top_rows = torch.topk(logits, k=k, dim=0)
        pairs = []
        for c in range(cold_t.shape[0]):
            iid = int(cold_t[c].item())
            for t in range(k):
                pairs.append((int(top_rows[t, c].item()), iid))
        R_hat = _add_edges_to_R(R_base, pairs)
        adj_hat = self.data.normalize_graph_mat(_bip_adj_from_R(R_hat, n_users, n_items))
        adj_t = _sparse_adj_tensor(adj_hat, self.device)
        z_u, z_i = _lightgcn_mean_all_layers(adj_t, model.user_emb, x_all, n_users, self.Lh)
        return torch.matmul(z_u, z_i.transpose(0, 1))

    def predict(self, u):
        with torch.no_grad():
            scores = self.batch_predict([u])
            return scores[0].cpu().numpy()

    def batch_predict(self, users):
        with torch.no_grad():
            model = self.model.to(self.device)
            if self.best_state_dict is not None:
                model.load_state_dict({k: v.to(self.device) for k, v in self.best_state_dict.items()})
            scores_full = self._forward_scores_ghat(model)
            u_list = self.data.get_user_id_list(users)
            u_list = torch.tensor(u_list, dtype=torch.long, device=self.device)
            return scores_full[u_list]

"""
FS-GNN (Feature-Structure Adaptive Completion GNN) for ColdRec.

Aligned with Lei et al., AAAI 2025, under ColdRec constraints:
- Initial node features: preprocessed content (+ Linear), replacing LLM+PCA.
- Feature completion: GAT encoder (Eq.4–6): LeakyReLU(0.2), softmax attention,
  aggregate alpha * h0 (neighbor) in embedding space; denoise dropout on cold side only;
  learnable GCN decoder: stacked (Ã X W) with ReLU between layers (Kipf-style);
  L_fc = MSE(dec, h0) on cold-start nodes only by default (mapped_cold_*).
- Structure completion: PPR top-k (optional PPR mass as edge weight); cosine kNN with
  similarity as edge weight (Eq.9–11); multi-layer GCN + MLP edge loss L_sc.
- Merged graph (Eq.15) with weighted same-type block + U–I block; LightGCN (Eq.16–17).
- L_pred: BPR (+ L2) for ColdRec implicit top-K evaluation (paper uses MSE on ratings).

Only this file and config/model_param.py (FSGNN args) are touched for FS-GNN.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.databuilder import TorchGraphInterface
from util.utils import bpr_loss, l2_reg_loss, next_batch_pairwise

from .BaseRecommender import BaseColdStartTrainer


def _normalize_graph_mat(adj_mat: sp.spmatrix) -> sp.csr_matrix:
    adj_mat = adj_mat.tocsr()
    rowsum = np.array(adj_mat.sum(1)).flatten()
    d_inv = np.zeros_like(rowsum, dtype=np.float32)
    np.power(rowsum, -0.5, out=d_inv, where=rowsum != 0)
    d_mat_inv = sp.diags(d_inv)
    norm_adj_tmp = d_mat_inv.dot(adj_mat)
    return norm_adj_tmp.dot(d_mat_inv).tocsr()


def _row_stochastic_P(adj_with_self: sp.csr_matrix) -> sp.csr_matrix:
    adj_with_self = adj_with_self.tocsr()
    deg = np.array(adj_with_self.sum(1)).flatten()
    d_inv = np.zeros_like(deg, dtype=np.float64)
    np.divide(1.0, deg, out=d_inv, where=deg > 0)
    d_mat_inv = sp.diags(d_inv)
    return (d_mat_inv @ adj_with_self).tocsr()


def _ppr_topk_extra_edges(
    r_train: sp.csr_matrix,
    delta: float,
    n_iter: int,
    topk: int,
    weighted: bool,
) -> sp.csr_matrix:
    """PPR restart-to-user; add top-k item links per user (exclude observed). Optionally weight by PPR mass."""
    nu, ni = r_train.shape
    n = nu + ni
    rows, cols = r_train.nonzero()
    row_idx = np.concatenate([rows, cols + nu])
    col_idx = np.concatenate([cols + nu, rows])
    data = np.ones(len(row_idx), dtype=np.float32)
    a_hat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n, n), dtype=np.float32).tocsr()
    a_hat = a_hat + sp.eye(n, format="csr", dtype=np.float32)
    p_mat = _row_stochastic_P(a_hat).astype(np.float64)

    pi = np.zeros((nu, n), dtype=np.float64)
    pi[np.arange(nu), np.arange(nu)] = 1.0
    restart = np.zeros((nu, n), dtype=np.float64)
    restart[np.arange(nu), np.arange(nu)] = float(delta)
    one_m = 1.0 - float(delta)
    for _ in range(int(n_iter)):
        pi = one_m * p_mat.transpose().dot(pi.transpose()).transpose() + restart

    scores = pi[:, nu:]
    r_train = r_train.tocsr()
    extra = sp.lil_matrix((nu, ni), dtype=np.float32)
    for u in range(nu):
        observed = r_train[u].indices if r_train[u].nnz else np.array([], dtype=np.int32)
        n_obs = len(observed)
        if ni <= n_obs:
            continue
        cand = scores[u].copy()
        cand[observed] = -np.inf
        k_eff = min(int(topk), ni - n_obs)
        if k_eff <= 0:
            continue
        top = np.argpartition(-cand, kth=k_eff - 1)[:k_eff]
        if weighted:
            w = cand[top].astype(np.float32)
            w = np.clip(w, 0.0, None)
            max_w = float(np.max(w)) if w.size else 0.0
            if max_w > 0:
                w = w / max_w
            extra[u, top] = w
        else:
            extra[u, top] = 1.0
    return extra.tocsr()


def _cosine_knn_graph(
    feat: np.ndarray, k: int, weighted: bool, row_chunk: int = 256
) -> sp.csr_matrix:
    """
    kNN on cosine similarity (Eq.9–11). If weighted, edge (m,n) carries S_mn; symmetrize by maximum.
    """
    n = feat.shape[0]
    if n <= 1:
        return sp.csr_matrix((n, n), dtype=np.float32)
    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        return sp.csr_matrix((n, n), dtype=np.float32)

    x = feat.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms

    rows_l, cols_l, vals_l = [], [], []
    for i0 in range(0, n, row_chunk):
        i1 = min(i0 + row_chunk, n)
        sim = x[i0:i1] @ x.T
        for r in range(i1 - i0):
            sim[r, i0 + r] = -np.inf
        take = min(k_eff, n - 1)
        part = np.argpartition(-sim, kth=take - 1, axis=1)[:, :take]
        for r, global_i in enumerate(range(i0, i1)):
            for j in part[r]:
                j = int(j)
                if j == global_i:
                    continue
                s = float(sim[r, j])
                if not weighted:
                    s = 1.0
                else:
                    # Cosine in [-1,1] -> nonnegative edge weight in (0,1] for GCN
                    s = max(1e-6, min(1.0, (s + 1.0) * 0.5))
                rows_l.append(global_i)
                cols_l.append(j)
                vals_l.append(s)
    if not rows_l:
        return sp.csr_matrix((n, n), dtype=np.float32)
    knn = sp.coo_matrix((vals_l, (rows_l, cols_l)), shape=(n, n), dtype=np.float32).tocsr()
    knn = knn.maximum(knn.T)
    knn.setdiag(0)
    knn.eliminate_zeros()
    return knn


def _merge_block_weighted(
    r_ui: sp.csr_matrix, a_uu: sp.csr_matrix, a_ii: sp.csr_matrix
) -> sp.csr_matrix:
    """Eq.15: [[A_uu, R],[R^T, A_ii]] with nonnegative weights; symmetrize."""
    nu, ni = r_ui.shape
    r_ui = r_ui.tocsr().astype(np.float32)
    a_uu = a_uu.tocsr().astype(np.float32)
    a_ii = a_ii.tocsr().astype(np.float32)
    z_uu = sp.csr_matrix((nu, nu), dtype=np.float32)
    z_ii = sp.csr_matrix((ni, ni), dtype=np.float32)
    r_block = sp.bmat([[z_uu, r_ui], [r_ui.T, z_ii]], format="csr")
    a_block = sp.block_diag((a_uu, a_ii), format="csr")
    adj = (a_block + r_block).tocsr()
    adj = adj.maximum(adj.T)
    adj.eliminate_zeros()
    return adj


def _load_side_features(
    data, args, n_nodes: int, side: str, fallback_dim: int
) -> Tuple[torch.Tensor, int]:
    path = os.path.join(".", "data", args.dataset, f"{args.dataset}_{side}_content.npy")
    arr: Optional[np.ndarray] = None
    if side == "user" and getattr(data, "mapped_user_content", None) is not None:
        arr = np.asarray(data.mapped_user_content, dtype=np.float32)
    elif side == "item" and getattr(data, "mapped_item_content", None) is not None:
        arr = np.asarray(data.mapped_item_content, dtype=np.float32)
    elif os.path.isfile(path):
        arr = np.load(path).astype(np.float32)
    if arr is None:
        dim = int(fallback_dim)
        t = torch.randn(n_nodes, dim, dtype=torch.float32) * 0.01
        return t, dim
    if arr.shape[0] != n_nodes:
        raise ValueError(
            f"FSGNN: {side}_content row count {arr.shape[0]} != {n_nodes} ({side}s in graph)."
        )
    return torch.from_numpy(arr), int(arr.shape[1])


class FSGNN_Learner(nn.Module):
    def __init__(self, args, data, emb_size: int, device: torch.device):
        super().__init__()
        self.args = args
        self.data = data
        self.emb_size = emb_size
        self.device = device

        nu, ni = data.user_num, data.item_num
        raw_u, du = _load_side_features(data, args, nu, "user", args.fsgnn_fallback_dim)
        raw_i, di = _load_side_features(data, args, ni, "item", args.fsgnn_fallback_dim)
        self.register_buffer("raw_u", raw_u, persistent=False)
        self.register_buffer("raw_i", raw_i, persistent=False)

        self.proj_u = nn.Linear(du, emb_size)
        self.proj_i = nn.Linear(di, emb_size)

        hid = int(args.fsgnn_gat_hidden)
        self.lin_u_gat = nn.Linear(emb_size, hid)
        self.lin_i_gat = nn.Linear(emb_size, hid)
        self.gat_att = nn.Linear(2 * hid, 1, bias=False)
        self._gat_leaky = 0.2

        self.dropout_p = float(args.fsgnn_p_drop)
        self.dropout_cold_only = bool(args.fsgnn_dropout_cold_side_only)
        self.lfc_cold_only = bool(args.fsgnn_lfc_cold_side_only)

        ci = np.asarray(data.mapped_cold_item_idx, dtype=np.int64)
        cu = np.asarray(data.mapped_cold_user_idx, dtype=np.int64)
        self.register_buffer("cold_item_idx", torch.from_numpy(ci), persistent=False)
        self.register_buffer("cold_user_idx", torch.from_numpy(cu), persistent=False)

        n_fc_dec = max(1, int(args.fsgnn_fc_decoder_layers))
        self.fc_dec = nn.ModuleList(
            [nn.Linear(emb_size, emb_size, bias=True) for _ in range(n_fc_dec)]
        )

        r_train = data.interaction_mat.tocsr()
        r_extra = _ppr_topk_extra_edges(
            r_train,
            delta=float(args.fsgnn_ppr_delta),
            n_iter=int(args.fsgnn_ppr_iter),
            topk=int(args.fsgnn_ppr_topk),
            weighted=bool(args.fsgnn_ppr_weighted),
        )
        r_ui = (r_train.astype(np.float32) + r_extra).tocsr()
        r_ui.eliminate_zeros()

        fu = raw_u.cpu().numpy()
        fi = raw_i.cpu().numpy()
        a_uu = _cosine_knn_graph(fu, int(args.fsgnn_knn_k), weighted=bool(args.fsgnn_knn_weighted))
        a_ii = _cosine_knn_graph(fi, int(args.fsgnn_knn_k), weighted=bool(args.fsgnn_knn_weighted))

        adj_complete = _merge_block_weighted(r_ui, a_uu, a_ii)
        self.register_buffer(
            "adj_complete",
            TorchGraphInterface.convert_sparse_mat_to_tensor(_normalize_graph_mat(adj_complete)),
            persistent=False,
        )

        adj_train_bip = data.create_sparse_complete_bipartite_adjacency(self_connection=False)
        self.register_buffer(
            "adj_train",
            TorchGraphInterface.convert_sparse_mat_to_tensor(data.normalize_graph_mat(adj_train_bip)),
            persistent=False,
        )

        a_uu_looped = (a_uu + sp.eye(nu, format="csr", dtype=np.float32)).tocsr()
        a_ii_looped = (a_ii + sp.eye(ni, format="csr", dtype=np.float32)).tocsr()
        self.register_buffer(
            "adj_uu",
            TorchGraphInterface.convert_sparse_mat_to_tensor(_normalize_graph_mat(a_uu_looped)),
            persistent=False,
        )
        self.register_buffer(
            "adj_ii",
            TorchGraphInterface.convert_sparse_mat_to_tensor(_normalize_graph_mat(a_ii_looped)),
            persistent=False,
        )

        ru, ci = r_train.nonzero()
        self.register_buffer("edge_u", torch.from_numpy(ru.astype(np.int64)), persistent=False)
        self.register_buffer("edge_i", torch.from_numpy(ci.astype(np.int64)), persistent=False)

        uu_r, uu_c = sp.triu(a_uu, k=1).nonzero()
        self.register_buffer("uu_row", torch.from_numpy(uu_r.astype(np.int64)), persistent=False)
        self.register_buffer("uu_col", torch.from_numpy(uu_c.astype(np.int64)), persistent=False)
        ii_r, ii_c = sp.triu(a_ii, k=1).nonzero()
        self.register_buffer("ii_row", torch.from_numpy(ii_r.astype(np.int64)), persistent=False)
        self.register_buffer("ii_col", torch.from_numpy(ii_c.astype(np.int64)), persistent=False)

        self.struct_mlp = nn.Sequential(
            nn.Linear(2 * emb_size, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        n_sc = max(1, int(args.fsgnn_sc_layers))
        self.sc_uu = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=True) for _ in range(n_sc)])
        self.sc_ii = nn.ModuleList([nn.Linear(emb_size, emb_size, bias=True) for _ in range(n_sc)])

        self.id_residual = bool(args.fsgnn_id_residual)
        if self.id_residual:
            self.emb_user = nn.Parameter(torch.empty(nu, emb_size))
            self.emb_item = nn.Parameter(torch.empty(ni, emb_size))
            nn.init.xavier_uniform_(self.emb_user)
            nn.init.xavier_uniform_(self.emb_item)
        else:
            self.register_parameter("emb_user", None)
            self.register_parameter("emb_item", None)

        self.n_layers = int(args.fsgnn_light_layers)

    def _apply_feature_dropout(self, h0_u: torch.Tensor, h0_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.dropout_p
        if p <= 0:
            return h0_u, h0_i
        tr = self.training
        if self.dropout_cold_only:
            if self.args.cold_object == "item":
                return h0_u, F.dropout(h0_i, p, training=tr)
            if self.args.cold_object == "user":
                return F.dropout(h0_u, p, training=tr), h0_i
        return F.dropout(h0_u, p, training=tr), F.dropout(h0_i, p, training=tr)

    def _bipartite_gat_encoder(
        self, h0_u: torch.Tensor, h0_i: torch.Tensor, hu_t: torch.Tensor, hi_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eq.4–6: attention from LeakyReLU(a^T[W e_i || W e_u]); encode with sum_u alpha_iu * e_u + tilde e_i.
        """
        nu, ni = h0_u.shape[0], h0_i.shape[0]
        eu, ei = self.edge_u, self.edge_i
        d = h0_u.size(-1)
        if eu.numel() == 0:
            return torch.sigmoid(hu_t), torch.sigmoid(hi_t)

        wh_u = self.lin_u_gat(h0_u)
        wh_i = self.lin_i_gat(h0_i)
        z_i = self.gat_att(torch.cat([wh_i[ei], wh_u[eu]], dim=-1)).squeeze(-1)
        z_i = F.leaky_relu(z_i, negative_slope=self._gat_leaky)
        alpha_i = torch.exp(z_i.clamp(-20.0, 20.0))
        den_i = torch.zeros(ni, device=h0_u.device, dtype=h0_u.dtype)
        den_i.index_add_(0, ei, alpha_i)
        den_e = den_i[ei]
        w = alpha_i / den_e.clamp_min(1e-8)
        w = w * (den_e > 1e-8).to(w.dtype)
        agg_i = torch.zeros(ni, d, device=h0_u.device, dtype=h0_u.dtype)
        agg_i.index_add_(0, ei, w.unsqueeze(1) * h0_u[eu])
        enc_i = torch.sigmoid(agg_i + hi_t)

        z_u = self.gat_att(torch.cat([wh_u[eu], wh_i[ei]], dim=-1)).squeeze(-1)
        z_u = F.leaky_relu(z_u, negative_slope=self._gat_leaky)
        alpha_u = torch.exp(z_u.clamp(-20.0, 20.0))
        den_u = torch.zeros(nu, device=h0_u.device, dtype=h0_u.dtype)
        den_u.index_add_(0, eu, alpha_u)
        den_eu = den_u[eu]
        wu = alpha_u / den_eu.clamp_min(1e-8)
        wu = wu * (den_eu > 1e-8).to(wu.dtype)
        agg_u = torch.zeros(nu, d, device=h0_u.device, dtype=h0_u.dtype)
        agg_u.index_add_(0, eu, wu.unsqueeze(1) * h0_i[ei])
        enc_u = torch.sigmoid(agg_u + hu_t)
        return enc_u, enc_i

    def _gcn_decode(self, enc: torch.Tensor) -> torch.Tensor:
        """Kipf GCN layer: H' = σ(Ã H W); last layer has no σ (reconstruction to h0)."""
        x = enc
        for k, lin in enumerate(self.fc_dec):
            x = lin(x)
            x = torch.sparse.mm(self.adj_train, x)
            if k < len(self.fc_dec) - 1:
                x = F.relu(x)
        return x

    def _lfc_loss(
        self, dec_u: torch.Tensor, dec_i: torch.Tensor, h0_u: torch.Tensor, h0_i: torch.Tensor
    ) -> torch.Tensor:
        """Paper-style L_fc on reconstructed vs projected content; optional cold-side only."""
        if not self.lfc_cold_only:
            return F.mse_loss(dec_u, h0_u) + F.mse_loss(dec_i, h0_i)

        if self.args.cold_object == "item":
            idx = self.cold_item_idx
            if idx.numel() == 0:
                return F.mse_loss(dec_u, h0_u) + F.mse_loss(dec_i, h0_i)
            return F.mse_loss(dec_i[idx], h0_i[idx])

        if self.args.cold_object == "user":
            idx = self.cold_user_idx
            if idx.numel() == 0:
                return F.mse_loss(dec_u, h0_u) + F.mse_loss(dec_i, h0_i)
            return F.mse_loss(dec_u[idx], h0_u[idx])

        return F.mse_loss(dec_u, h0_u) + F.mse_loss(dec_i, h0_i)

    def _feature_completion(
        self, h0_u: torch.Tensor, h0_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hu_t, hi_t = self._apply_feature_dropout(h0_u, h0_i)
        enc_u, enc_i = self._bipartite_gat_encoder(h0_u, h0_i, hu_t, hi_t)
        enc = torch.cat([enc_u, enc_i], dim=0)
        dec = self._gcn_decode(enc)
        nu = h0_u.shape[0]
        dec_u, dec_i = dec[:nu], dec[nu:]
        l_fc = self._lfc_loss(dec_u, dec_i, h0_u, h0_i)
        return dec_u, dec_i, l_fc

    def _structure_gcn(self, h: torch.Tensor, adj: torch.Tensor, layers: nn.ModuleList) -> torch.Tensor:
        x = h
        for lin in layers:
            x = F.relu(torch.sparse.mm(adj, lin(x)))
        return x

    def _structure_loss(self, hu: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        hu_n = self._structure_gcn(hu, self.adj_uu, self.sc_uu)
        hi_n = self._structure_gcn(hi, self.adj_ii, self.sc_ii)

        max_e = int(self.args.fsgnn_sc_max_edges)
        dev = hu.device
        l_parts = []

        if self.uu_row.numel() > 0:
            m = self.uu_row.numel()
            sel = torch.randperm(m, device=dev)[: min(m, max_e)] if m > max_e else torch.arange(m, device=dev)
            a, b = self.uu_row[sel], self.uu_col[sel]
            pred = self.struct_mlp(torch.cat([hu_n[a], hu_n[b]], dim=-1)).squeeze(-1)
            l_parts.append(F.mse_loss(pred, torch.ones_like(pred)))

        if self.ii_row.numel() > 0:
            m = self.ii_row.numel()
            sel = torch.randperm(m, device=dev)[: min(m, max_e)] if m > max_e else torch.arange(m, device=dev)
            a, b = self.ii_row[sel], self.ii_col[sel]
            pred = self.struct_mlp(torch.cat([hi_n[a], hi_n[b]], dim=-1)).squeeze(-1)
            l_parts.append(F.mse_loss(pred, torch.ones_like(pred)))

        if not l_parts:
            return torch.zeros((), device=dev, dtype=hu.dtype)
        return sum(l_parts) / len(l_parts)

    def _lightgcn(self, ego_u: torch.Tensor, ego_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat([ego_u, ego_i], dim=0)
        embs = [ego]
        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj_complete, x)
            embs.append(x)
        out = torch.stack(embs, dim=0).mean(dim=0)
        nu = ego_u.shape[0]
        return out[:nu], out[nu:]

    def forward_training(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_u = self.raw_u.to(self.device)
        raw_i = self.raw_i.to(self.device)
        h0_u = self.proj_u(raw_u)
        h0_i = self.proj_i(raw_i)
        dec_u, dec_i, l_fc = self._feature_completion(h0_u, h0_i)
        l_sc = self._structure_loss(dec_u, dec_i)
        if self.id_residual:
            ego_u = dec_u + self.emb_user
            ego_i = dec_i + self.emb_item
        else:
            ego_u, ego_i = dec_u, dec_i
        u_emb, i_emb = self._lightgcn(ego_u, ego_i)
        return u_emb, i_emb, l_fc, l_sc

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u_emb, i_emb, _, _ = self.forward_training()
        return u_emb, i_emb


class FSGNN(BaseColdStartTrainer):
    def __init__(self, config):
        super(FSGNN, self).__init__(config)
        self.model = FSGNN_Learner(self.args, self.data, self.emb_size, self.device)

    def train(self):
        model = self.model.to(self.device)
        wd = float(self.args.fsgnn_weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=wd)
        lam = float(self.args.fsgnn_lambda_fc)
        mu = float(self.args.fsgnn_mu_sc)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.LongTensor(user_idx).to(self.device)
                pos_idx = torch.LongTensor(pos_idx).to(self.device)
                neg_idx = torch.LongTensor(neg_idx).to(self.device)

                rec_u, rec_i, l_fc, l_sc = model.forward_training()
                u_e, p_e, n_e = rec_u[user_idx], rec_i[pos_idx], rec_i[neg_idx]
                batch_loss = (
                    bpr_loss(u_e, p_e, n_e)
                    + lam * l_fc
                    + mu * l_sc
                    + l2_reg_loss(self.reg, u_e, p_e, n_e)
                )
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print(
                        "training:",
                        epoch + 1,
                        "batch",
                        n,
                        "batch_loss:",
                        batch_loss.item(),
                        "l_fc:",
                        l_fc.item(),
                        "l_sc:",
                        l_sc.item(),
                    )

            with torch.no_grad():
                model.eval()
                self.user_emb, self.item_emb = model()
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type="all")
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.timer(start=False)
        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(
                self.user_emb,
                f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt",
            )
            torch.save(
                self.item_emb,
                f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt",
            )

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

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

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseRecommender import BaseColdStartTrainer
from util.utils import bpr_loss, l2_reg_loss, next_batch_pairwise


class LightCSR(BaseColdStartTrainer):
    """Standalone implementation of the final LightCSR adapter.

    ``cmct_warm_beta=0`` keeps warm item embeddings exactly locked to the
    backbone, while positive values enable bounded warm-side materialization.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.args.cold_object != 'item':
            raise ValueError('LightCSR currently supports strict cold-item recommendation only.')

        user_path = f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt'
        item_path = f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt'
        user_init = torch.load(user_path, map_location='cpu').float()
        item_init = torch.load(item_path, map_location='cpu').float()
        if user_init.shape[1] != self.emb_size or item_init.shape[1] != self.emb_size:
            raise ValueError(
                f'Backbone embedding dim mismatch: expected {self.emb_size}, '
                f'got user={tuple(user_init.shape)}, item={tuple(item_init.shape)}.'
            )

        freeze = bool(getattr(self.args, 'cmct_freeze_backbone', True))
        self.user_param = nn.Parameter(user_init, requires_grad=not freeze)
        self.item_param = nn.Parameter(item_init, requires_grad=not freeze)

        self.model = LightCSRLearner(self.args, self.data, self.emb_size)
        self.item_content = torch.tensor(
            self.data.mapped_item_content, dtype=torch.float32, device=self.device
        )
        self.warm_ids = torch.tensor(self.data.mapped_warm_item_idx, dtype=torch.long, device=self.device)
        self.cold_ids = torch.tensor(self.data.mapped_cold_item_idx, dtype=torch.long, device=self.device)

        self._item_to_cold_pos = torch.full((self.data.item_num,), -1, dtype=torch.long, device=self.device)
        self._item_to_cold_pos[self.cold_ids] = torch.arange(len(self.cold_ids), device=self.device)

        warm_ids_np = np.asarray(self.data.mapped_warm_item_idx, dtype=np.int64)
        cold_ids_np = np.asarray(self.data.mapped_cold_item_idx, dtype=np.int64)
        content = np.asarray(self.data.mapped_item_content, dtype=np.float32)
        k = int(getattr(self.args, 'cmct_knn_k', 10))
        chunk = int(getattr(self.args, 'cmct_knn_chunk', 1024))
        temp = float(getattr(self.args, 'cmct_kernel_temp', 0.2))
        warm_knn, warm_w = _content_knn(
            content[warm_ids_np], content[warm_ids_np], warm_ids_np, k, temp, chunk, exclude_self=True
        )
        cold_knn, cold_w = _content_knn(
            content[cold_ids_np], content[warm_ids_np], warm_ids_np, k, temp, chunk, exclude_self=False
        )
        self.warm_knn_ids = torch.tensor(warm_knn, dtype=torch.long, device=self.device)
        self.warm_knn_w = torch.tensor(warm_w, dtype=torch.float32, device=self.device)
        self.cold_knn_ids = torch.tensor(cold_knn, dtype=torch.long, device=self.device)
        self.cold_knn_w = torch.tensor(cold_w, dtype=torch.float32, device=self.device)

        self._item_pos_users, self._item_pos_user_sets = self._build_item_positive_users()

    def train(self):
        self.user_param.data = self.user_param.data.to(self.device)
        self.item_param.data = self.item_param.data.to(self.device)
        model = self.model.to(self.device)

        params = list(model.parameters())
        if self.user_param.requires_grad:
            params.append(self.user_param)
        if self.item_param.requires_grad:
            params.append(self.item_param)
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=float(getattr(self.args, 'cmct_weight_decay', 0.0)),
        )

        self.timer(start=True)
        epoch = -1
        for epoch in range(self.maxEpoch):
            model.train()
            cold_epoch_emb = self._generate_cold(detach_anchor=True).detach()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.tensor(user_idx, dtype=torch.long, device=self.device)
                pos_idx = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                neg_idx = torch.tensor(neg_idx, dtype=torch.long, device=self.device)

                user_emb = self.user_param[user_idx]
                pos_emb = self.item_param[pos_idx]
                neg_emb = self._batch_item_emb(neg_idx, cold_epoch_emb)
                loss = bpr_loss(user_emb, pos_emb, neg_emb)
                loss = loss + l2_reg_loss(self.reg, user_emb, pos_emb, neg_emb)

                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    clip = float(getattr(self.args, 'cmct_clip_norm', 5.0))
                    if clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, clip)
                    optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', loss.item())

            align_loss = self._alignment_step(optimizer, params)
            print('training:', epoch + 1, 'cmct_alignment_loss:', float(align_loss))

            with torch.no_grad():
                model.eval()
                self.user_emb, self.item_emb = self._materialize_embeddings()
                if epoch % self.eval_every == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag and self.early_stop_patience <= 0:
                        break

        self.epochs_ran = (epoch + 1) if self.maxEpoch > 0 else 0
        self.timer(start=False)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(
                self.user_emb,
                f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt',
            )
            torch.save(
                self.item_emb,
                f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt',
            )

    def _batch_item_emb(self, item_idx: torch.Tensor, cold_epoch_emb: torch.Tensor) -> torch.Tensor:
        emb = self.item_param[item_idx]
        cold_pos = self._item_to_cold_pos[item_idx]
        cold_mask = cold_pos >= 0
        if cold_mask.any():
            emb = emb.clone()
            emb[cold_mask] = cold_epoch_emb[cold_pos[cold_mask]]
        return emb

    def _build_item_positive_users(self):
        pos_users = [[] for _ in range(self.data.item_num)]
        pos_sets = [set() for _ in range(self.data.item_num)]
        for raw_item, raw_users in self.data.training_set_i.items():
            iid = self.data.item[raw_item]
            users = [self.data.user[u] for u in raw_users.keys()]
            pos_users[iid] = users
            pos_sets[iid] = set(users)
        tensors = []
        for users in pos_users:
            if users:
                tensors.append(torch.tensor(users, dtype=torch.long, device=self.device))
            else:
                tensors.append(torch.empty(0, dtype=torch.long, device=self.device))
        return tensors, pos_sets

    def _build_user_positive_item_sets(self):
        if hasattr(self, '_ca_user_pos_items'):
            return self._ca_user_pos_items
        pos_sets = [set() for _ in range(self.data.user_num)]
        for raw_user, raw_items in self.data.training_set_u.items():
            uid = self.data.user[raw_user]
            pos_sets[uid] = {self.data.item[item] for item in raw_items.keys()}
        self._ca_user_pos_items = pos_sets
        return pos_sets

    def _alignment_components(
        self,
        target_ids: torch.Tensor,
        knn_ids: torch.Tensor,
        weights: torch.Tensor,
        detach_anchor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen, knn, _ = self._generate(target_ids, knn_ids, weights, detach_anchor)
        teacher = self.model.encoder(self.item_content[target_ids])
        return gen, knn, teacher

    def _alignment_step(self, optimizer, params) -> torch.Tensor:
        align_bs = int(getattr(self.args, 'cmct_align_batch', 4096))
        n_warm = len(self.warm_ids)
        if align_bs <= 0 or align_bs >= n_warm:
            pos = torch.arange(n_warm, device=self.device)
        else:
            pos = torch.randperm(n_warm, device=self.device)[:align_bs]

        target_ids = self.warm_ids[pos]
        knn_ids = self.warm_knn_ids[pos]
        knn_weights = self.warm_knn_w[pos]
        gen, knn, teacher = self._alignment_components(
            target_ids,
            knn_ids,
            knn_weights,
            detach_anchor=True,
        )
        target = self.item_param[target_ids].detach()

        embed_weight = float(getattr(self.args, 'cmct_embed_weight', 1.0))
        loss = gen.new_tensor(0.0)
        if embed_weight > 0:
            loss = loss + embed_weight * F.mse_loss(gen, target)

        teacher_weight = float(getattr(self.args, 'dct_teacher_weight', 0.5))
        if teacher_weight > 0 and teacher is not None:
            loss = loss + teacher_weight * F.mse_loss(teacher, target)

        score_weight = float(getattr(self.args, 'cmct_score_weight', 0.1))
        if score_weight > 0:
            loss = loss + score_weight * self._score_alignment_loss(gen, target)
        loss = loss + self._positive_user_score_distill(target_ids, gen, target)

        pair_weight = float(getattr(self.args, 'cmct_pair_weight', 0.0))
        if pair_weight > 0:
            loss = loss + pair_weight * self._pairwise_rank_loss(target_ids, gen)

        loss = loss + self._candidate_item_rank_loss(target_ids, gen)
        loss = loss + self._local_hard_contrast_loss(target_ids, gen, knn_ids)
        loss = loss + self._candidate_distribution_distill_loss(
            target_ids,
            gen,
            target,
            knn_ids,
            knn_weights,
        )

        teacher_lc_weight = float(getattr(self.args, 'clt_teacher_lc_weight', 0.0))
        if teacher_lc_weight > 0 and teacher is not None:
            temp = max(float(getattr(self.args, 'lc_temp', 0.2)), 1e-6)
            n_hard = int(getattr(self.args, 'lc_hard_negatives', 8))
            loss = loss + teacher_lc_weight * self._local_hard_contrast(target_ids, teacher, knn_ids, temp, n_hard)

        cl_weight = float(getattr(self.args, 'cmct_contrast_weight', 0.1))
        if cl_weight > 0 and gen.shape[0] > 1:
            temp = float(getattr(self.args, 'cmct_contrast_temp', 0.2))
            logits = F.normalize(gen, dim=1) @ F.normalize(target, dim=1).T / temp
            labels = torch.arange(gen.shape[0], device=self.device)
            loss = loss + cl_weight * F.cross_entropy(logits, labels)

        teacher_cl_weight = float(getattr(self.args, 'dct_teacher_contrast_weight', 0.0))
        if teacher_cl_weight > 0 and teacher is not None and teacher.shape[0] > 1:
            temp = float(getattr(self.args, 'cmct_contrast_temp', 0.2))
            logits = F.normalize(teacher, dim=1) @ F.normalize(target, dim=1).T / temp
            labels = torch.arange(teacher.shape[0], device=self.device)
            loss = loss + teacher_cl_weight * F.cross_entropy(logits, labels)

        lap_weight = float(getattr(self.args, 'cmct_lap_weight', 0.0))
        if lap_weight > 0:
            loss = loss + lap_weight * F.mse_loss(gen, knn.detach())

        optimizer.zero_grad()
        loss.backward()
        clip = float(getattr(self.args, 'cmct_clip_norm', 5.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        return loss.detach()

    def _generate_cold(self, detach_anchor: bool) -> torch.Tensor:
        gen, _, _ = self._generate(self.cold_ids, self.cold_knn_ids, self.cold_knn_w, detach_anchor)
        return gen

    def _generate_warm(self, detach_anchor: bool) -> torch.Tensor:
        gen, _, _ = self._generate(self.warm_ids, self.warm_knn_ids, self.warm_knn_w, detach_anchor)
        return gen

    def _generate(
        self,
        target_ids: torch.Tensor,
        knn_ids: torch.Tensor,
        weights: torch.Tensor,
        detach_anchor: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        teacher = self.model.encoder(self.item_content[target_ids])
        if bool(getattr(self.args, 'lightcsr_no_neighbor_message', False)):
            return teacher, teacher.detach() if detach_anchor else teacher, teacher

        anchors = self.item_param[knn_ids]
        if detach_anchor:
            anchors = anchors.detach()

        knn_emb = torch.sum(anchors * weights.unsqueeze(-1), dim=1)
        neigh_teacher = self._neighbor_teacher(knn_ids)
        local_residual = torch.sum((anchors - neigh_teacher) * weights.unsqueeze(-1), dim=1)

        max_res = float(getattr(self.args, 'drdh_max_residual_rel', 1.0))
        local_residual = self._bounded_delta(local_residual, knn_emb, max_res)
        dr_direct = teacher + local_residual

        rho = self._local_reliability(anchors, knn_emb, weights)
        min_mix = float(getattr(self.args, 'dh_min_direct_mix', 0.0))
        max_mix = float(getattr(self.args, 'dh_max_direct_mix', 0.75))
        if min_mix > max_mix:
            raise ValueError('dh_min_direct_mix must be <= dh_max_direct_mix.')
        power = float(getattr(self.args, 'dh_gate_power', 1.0))
        uncertainty = (1.0 - rho).clamp_min(1e-8)
        if power != 1.0:
            uncertainty = uncertainty.pow(power)
        direct_mix = (min_mix + (max_mix - min_mix) * uncertainty).clamp(0.0, 1.0)

        gen = (1.0 - direct_mix) * knn_emb + direct_mix * dr_direct
        gen = self._apply_norm_calibration(gen, knn_ids, weights, detach_anchor=detach_anchor)

        rel_anchors = self.item_param[knn_ids]
        if detach_anchor:
            rel_anchors = rel_anchors.detach()
        rel_knn = knn_emb.detach() if detach_anchor else knn_emb
        rho = self._local_reliability(rel_anchors, rel_knn, weights)
        gen = self._reliability_norm_calibration(gen, knn_ids, weights, rho, detach_anchor=detach_anchor)
        return gen, knn_emb, teacher

    def _neighbor_teacher(self, knn_ids: torch.Tensor) -> torch.Tensor:
        flat = knn_ids.reshape(-1)
        unique_ids, inv = torch.unique(flat, sorted=False, return_inverse=True)
        encoded = self.model.encoder(self.item_content[unique_ids])
        if bool(getattr(self.args, 'drdh_detach_neighbor_teacher', True)):
            encoded = encoded.detach()
        return encoded[inv].view(*knn_ids.shape, self.emb_size)

    def _bounded_delta(self, delta: torch.Tensor, ref: torch.Tensor, max_rel: float) -> torch.Tensor:
        if max_rel <= 0:
            return delta
        delta_norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
        ref_norm = ref.detach().norm(dim=1, keepdim=True).clamp_min(1e-8)
        scale = (max_rel * ref_norm / delta_norm).clamp(max=1.0)
        return delta * scale

    def _local_reliability(
        self,
        anchors: torch.Tensor,
        knn_emb: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        diff = anchors - knn_emb.unsqueeze(1)
        local_var = torch.sum(weights * torch.sum(diff * diff, dim=-1), dim=1, keepdim=True)
        local_var = torch.log1p(local_var / max(1, self.emb_size))

        entropy = -torch.sum(weights * torch.log(weights.clamp_min(1e-12)), dim=1, keepdim=True)
        entropy = entropy / math.log(max(2, weights.shape[1]))
        inv_conf = 1.0 - torch.max(weights, dim=1, keepdim=True).values

        rho = 1.0 / (1.0 + entropy + local_var + inv_conf)
        power = float(getattr(self.args, 'dct_reliability_power', 1.0))
        if power != 1.0:
            rho = rho.clamp_min(1e-8).pow(power)

        min_rho = float(getattr(self.args, 'dct_min_reliability', 0.05))
        max_rho = float(getattr(self.args, 'dct_max_reliability', 0.95))
        if min_rho > max_rho:
            raise ValueError('dct_min_reliability must be <= dct_max_reliability.')
        return rho.clamp(min_rho, max_rho)

    def _neighbor_norm(self, knn_ids: torch.Tensor, weights: torch.Tensor, detach_anchor: bool) -> torch.Tensor:
        anchors = self.item_param[knn_ids]
        if detach_anchor:
            anchors = anchors.detach()
        norms = anchors.norm(dim=-1).clamp_min(1e-8)
        return torch.sum(weights * norms, dim=1, keepdim=True)

    def _apply_norm_calibration(
        self,
        emb: torch.Tensor,
        knn_ids: torch.Tensor,
        weights: torch.Tensor,
        detach_anchor: bool,
    ) -> torch.Tensor:
        mix = float(getattr(self.args, 'cmct_norm_mix', 1.0))
        if mix <= 0:
            return emb
        raw_norm = emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        neighbor_norm = self._neighbor_norm(knn_ids, weights, detach_anchor=detach_anchor)
        target_norm = ((1.0 - mix) * raw_norm + mix * neighbor_norm).clamp_min(1e-8)
        return F.normalize(emb, dim=1) * target_norm

    def _reliability_norm_calibration(
        self,
        emb: torch.Tensor,
        knn_ids: torch.Tensor,
        weights: torch.Tensor,
        rho: torch.Tensor,
        detach_anchor: bool,
    ) -> torch.Tensor:
        mix = float(getattr(self.args, 'wu_norm_mix', 0.0))
        if mix <= 0:
            return emb
        raw_norm = emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        neighbor_norm = self._neighbor_norm(knn_ids, weights, detach_anchor=detach_anchor)
        uncertainty = (1.0 - rho).clamp(0.0, 1.0)
        power = float(getattr(self.args, 'wu_norm_power', 1.0))
        if power != 1.0:
            uncertainty = uncertainty.clamp_min(1e-8).pow(power)
        beta = (mix * uncertainty).clamp(0.0, 1.0)
        target_norm = ((1.0 - beta) * raw_norm + beta * neighbor_norm).clamp_min(1e-8)
        return F.normalize(emb, dim=1) * target_norm

    def _materialize_embeddings(self):
        user_emb = self.user_param.detach().clone()
        item_emb = self.item_param.detach().clone()
        cold_gen = self._generate_cold(detach_anchor=True).detach()
        cold_gen = cold_gen * float(getattr(self.args, 'cmct_cold_scale', 1.0))
        item_emb[self.cold_ids] = cold_gen

        warm_beta = float(getattr(self.args, 'cmct_warm_beta', getattr(self.args, 'cmct_warm_mix', 0.0)))
        if warm_beta > 0:
            warm_gen = self._generate_warm(detach_anchor=True).detach()
            item_emb[self.warm_ids] = (1.0 - warm_beta) * item_emb[self.warm_ids] + warm_beta * warm_gen
        return user_emb, item_emb

    def _sample_positive_users(self, target_ids: torch.Tensor) -> torch.Tensor:
        out = []
        for iid in target_ids.detach().cpu().tolist():
            users = self._item_pos_users[int(iid)]
            if users.numel() == 0:
                out.append(torch.randint(self.data.user_num, (1,), device=self.device))
            else:
                out.append(users[torch.randint(users.numel(), (1,), device=self.device)])
        return torch.cat(out, dim=0)

    def _sample_negative_users(self, target_ids: torch.Tensor) -> torch.Tensor:
        out = []
        for iid in target_ids.detach().cpu().tolist():
            positives = self._item_pos_user_sets[int(iid)]
            cand = int(torch.randint(self.data.user_num, (1,), device=self.device).item())
            for _ in range(20):
                if cand not in positives:
                    break
                cand = int(torch.randint(self.data.user_num, (1,), device=self.device).item())
            out.append(cand)
        return torch.tensor(out, dtype=torch.long, device=self.device)

    def _score_alignment_loss(self, gen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_sample = int(getattr(self.args, 'cmct_score_users', 256))
        if n_sample <= 0:
            return gen.new_tensor(0.0)
        n_sample = min(n_sample, self.data.user_num)
        users = torch.randperm(self.data.user_num, device=self.device)[:n_sample]
        user_emb = self.user_param[users].detach()
        diff = gen - target
        score_err = user_emb @ diff.T
        return score_err.pow(2).mean() / max(1, self.emb_size)

    def _pairwise_rank_loss(self, target_ids: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
        n_pair = int(getattr(self.args, 'cmct_pair_samples', 1024))
        if n_pair <= 0 or gen.shape[0] == 0:
            return gen.new_tensor(0.0)
        n_pair = min(n_pair, gen.shape[0])
        if n_pair < gen.shape[0]:
            pos = torch.randperm(gen.shape[0], device=self.device)[:n_pair]
            target_ids = target_ids[pos]
            gen = gen[pos]
        pos_users = self._sample_positive_users(target_ids)
        neg_users = self._sample_negative_users(target_ids)
        pos_score = torch.sum(self.user_param[pos_users].detach() * gen, dim=1)
        neg_score = torch.sum(self.user_param[neg_users].detach() * gen, dim=1)
        return F.softplus(-(pos_score - neg_score)).mean()

    def _sample_negative_warm_items(self, users: torch.Tensor, n_neg: int) -> torch.Tensor:
        pos_sets = self._build_user_positive_item_sets()
        warm = self.warm_ids
        out = torch.empty((users.numel(), n_neg), dtype=torch.long, device=self.device)
        warm_n = int(warm.numel())
        for row, uid_t in enumerate(users.detach().cpu().tolist()):
            positives = pos_sets[int(uid_t)]
            picked = []
            for _ in range(n_neg):
                neg = int(warm[int(torch.randint(warm_n, (1,), device=self.device).item())].item())
                for _ in range(30):
                    if neg not in positives:
                        break
                    neg = int(warm[int(torch.randint(warm_n, (1,), device=self.device).item())].item())
                picked.append(neg)
            out[row] = torch.tensor(picked, dtype=torch.long, device=self.device)
        return out

    def _candidate_item_rank_loss(self, target_ids: torch.Tensor, gen: torch.Tensor) -> torch.Tensor:
        weight = float(getattr(self.args, 'ca_weight', 0.0))
        if weight <= 0 or gen.shape[0] == 0:
            return gen.new_tensor(0.0)

        n_pair = min(int(getattr(self.args, 'ca_pair_samples', 1024)), gen.shape[0])
        if n_pair <= 0:
            return gen.new_tensor(0.0)
        if n_pair < gen.shape[0]:
            pos = torch.randperm(gen.shape[0], device=self.device)[:n_pair]
            target_ids = target_ids[pos]
            gen = gen[pos]

        users = self._sample_positive_users(target_ids)
        n_neg = max(1, int(getattr(self.args, 'ca_negatives', 4)))
        neg_items = self._sample_negative_warm_items(users, n_neg)
        user_emb = self.user_param[users].detach()
        neg_emb = self.item_param[neg_items].detach()

        pos_score = torch.sum(user_emb * gen, dim=1)
        neg_score = torch.sum(user_emb.unsqueeze(1) * neg_emb, dim=-1)
        temp = max(float(getattr(self.args, 'ca_temp', 1.0)), 1e-6)
        hard_neg = torch.logsumexp(neg_score / temp, dim=1) * temp
        return weight * (F.softplus((hard_neg - pos_score) / temp).mean() * temp)

    def _local_hard_contrast(
        self,
        target_ids: torch.Tensor,
        emb: torch.Tensor,
        knn_ids: torch.Tensor,
        temp: float,
        n_hard: int,
    ) -> torch.Tensor:
        if emb.shape[0] == 0 or n_hard <= 0:
            return emb.new_tensor(0.0)
        n_hard = min(n_hard, knn_ids.shape[1])
        q = F.normalize(emb, dim=1)
        pos = F.normalize(self.item_param[target_ids].detach(), dim=1)
        neg = F.normalize(self.item_param[knn_ids[:, :n_hard]].detach(), dim=2)
        pos_logit = torch.sum(q * pos, dim=1, keepdim=True)
        neg_logits = torch.sum(q.unsqueeze(1) * neg, dim=2)
        logits = torch.cat([pos_logit, neg_logits], dim=1) / temp
        labels = torch.zeros(emb.shape[0], dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)

    def _local_hard_contrast_loss(
        self,
        target_ids: torch.Tensor,
        gen: torch.Tensor,
        knn_ids: torch.Tensor,
    ) -> torch.Tensor:
        weight = float(getattr(self.args, 'lc_weight', 0.0))
        if weight <= 0:
            return gen.new_tensor(0.0)
        temp = max(float(getattr(self.args, 'lc_temp', 0.2)), 1e-6)
        n_hard = int(getattr(self.args, 'lc_hard_negatives', 8))
        return weight * self._local_hard_contrast(target_ids, gen, knn_ids, temp, n_hard)

    def _positive_user_score_distill(
        self,
        target_ids: torch.Tensor,
        gen: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        weight = float(getattr(self.args, 'wu_pos_score_weight', 0.0))
        if weight <= 0 or gen.shape[0] == 0:
            return gen.new_tensor(0.0)
        users = self._sample_positive_users(target_ids)
        user_emb = self.user_param[users].detach()
        score_err = torch.sum(user_emb * (gen - target), dim=1)
        return weight * score_err.pow(2).mean() / max(1, self.emb_size)

    def _candidate_distribution_distill_loss(
        self,
        target_ids: torch.Tensor,
        gen: torch.Tensor,
        target: torch.Tensor,
        knn_ids: torch.Tensor,
        knn_weights: torch.Tensor,
    ) -> torch.Tensor:
        weight = float(getattr(self.args, 'new_rank_distill_weight', 0.0))
        if weight <= 0 or gen.shape[0] == 0 or knn_ids.shape[1] == 0:
            return gen.new_tensor(0.0)

        n_pair = min(int(getattr(self.args, 'new_rank_distill_samples', 1024)), gen.shape[0])
        if n_pair <= 0:
            return gen.new_tensor(0.0)
        if n_pair < gen.shape[0]:
            pos = torch.randperm(gen.shape[0], device=self.device)[:n_pair]
            target_ids = target_ids[pos]
            gen = gen[pos]
            target = target[pos]
            knn_ids = knn_ids[pos]
            knn_weights = knn_weights[pos]

        n_cand = min(max(1, int(getattr(self.args, 'new_rank_distill_candidates', 8))), knn_ids.shape[1])
        cand_ids = knn_ids[:, :n_cand]
        users = self._sample_positive_users(target_ids)

        user_emb = self.user_param[users].detach()
        cand_emb = self.item_param[cand_ids].detach()
        cand_logits = torch.sum(user_emb.unsqueeze(1) * cand_emb, dim=-1)
        target_teacher = torch.sum(user_emb * target, dim=1, keepdim=True)
        target_student = torch.sum(user_emb * gen, dim=1, keepdim=True)

        teacher_logits = torch.cat([target_teacher, cand_logits], dim=1)
        student_logits = torch.cat([target_student, cand_logits], dim=1)
        temp = max(float(getattr(self.args, 'new_rank_distill_temp', 0.5)), 1e-6)
        teacher_prob = F.softmax(teacher_logits / temp, dim=1).detach()
        student_log_prob = F.log_softmax(student_logits / temp, dim=1)
        per_sample = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=1) * temp * temp

        gate = str(getattr(self.args, 'new_rank_distill_gate', 'none')).lower()
        if gate != 'none':
            if gate not in {'uncertainty', 'reliability'}:
                raise ValueError('new_rank_distill_gate must be one of none, uncertainty, reliability.')
            anchors = self.item_param[knn_ids].detach()
            knn_emb = torch.sum(anchors * knn_weights.unsqueeze(-1), dim=1)
            rho = self._local_reliability(anchors, knn_emb, knn_weights).detach().squeeze(1)
            sample_gate = (1.0 - rho) if gate == 'uncertainty' else rho
            power = float(getattr(self.args, 'new_rank_distill_gate_power', 1.0))
            if power != 1.0:
                sample_gate = sample_gate.clamp_min(1e-8).pow(power)
            per_sample = per_sample * sample_gate.clamp(0.0, 1.0)

        return weight * per_sample.mean()

    def _calibrate_cold_scores(self, scores: torch.Tensor) -> torch.Tensor:
        weight = float(getattr(self.args, 'cw_cal_weight', 0.0))
        if weight <= 0 or scores.numel() == 0:
            return scores

        warm_ids = self.warm_ids
        cold_ids = self.cold_ids
        if warm_ids.numel() == 0 or cold_ids.numel() == 0:
            return scores

        warm_scores = scores[:, warm_ids]
        cold_scores = scores[:, cold_ids]
        top_mult = max(1, int(getattr(self.args, 'cw_cal_topk_mult', 1)))
        k_warm = min(warm_scores.shape[1], max(1, self.max_N * top_mult))
        k_cold = min(cold_scores.shape[1], max(1, self.max_N * top_mult))

        warm_ref = torch.topk(warm_scores, k_warm, dim=1).values.mean(dim=1, keepdim=True)
        cold_ref = torch.topk(cold_scores, k_cold, dim=1).values.mean(dim=1, keepdim=True)
        delta = warm_ref - cold_ref

        max_shift = float(getattr(self.args, 'cw_cal_max_shift', 1.0))
        if max_shift > 0:
            score_scale = warm_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-8)
            delta = delta.clamp(-max_shift * score_scale, max_shift * score_scale)

        calibrated = scores.clone()
        calibrated[:, cold_ids] = calibrated[:, cold_ids] + weight * delta
        return calibrated

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self._materialize_embeddings()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1)).unsqueeze(0)
            score = self._calibrate_cold_scores(score).squeeze(0)
            return score.cpu().numpy()

    def batch_predict(self, users):
        with torch.no_grad():
            users = self.data.get_user_id_list(users)
            users = torch.tensor(users, dtype=torch.long, device=self.device)
            scores = torch.matmul(self.user_emb[users], self.item_emb.transpose(0, 1))
            return self._calibrate_cold_scores(scores)


class LightCSRLearner(nn.Module):
    def __init__(self, args, data, emb_size):
        super().__init__()
        hidden = int(getattr(args, 'cmct_hidden', 256))
        dropout = float(getattr(args, 'cmct_dropout', 0.0))
        layers = int(getattr(args, 'cmct_encoder_layers', 2))
        content_dim = int(data.item_content_dim)

        modules = []
        in_dim = content_dim
        for _ in range(max(0, layers - 1)):
            modules.append(nn.Linear(in_dim, hidden))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            in_dim = hidden
        modules.append(nn.Linear(in_dim, emb_size))
        self.encoder_net = nn.Sequential(*modules)
        self.gate = nn.Linear(3, 1)
        nn.init.constant_(self.gate.bias, float(getattr(args, 'cmct_gate_bias_init', -2.0)))
        nn.init.constant_(self.gate.weight, 1.0)

    def encoder(self, content: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(content)


def _content_knn(
    query: np.ndarray,
    value: np.ndarray,
    value_ids: np.ndarray,
    k: int,
    temp: float,
    chunk_size: int,
    exclude_self: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    query_t = F.normalize(torch.from_numpy(query).float(), dim=1)
    value_t = F.normalize(torch.from_numpy(value).float(), dim=1)
    k_search = min(value_t.shape[0], k + 1 if exclude_self else k)
    all_ids = []
    all_weights = []
    for start in range(0, query_t.shape[0], max(1, chunk_size)):
        sims = query_t[start:start + chunk_size] @ value_t.T
        vals, idx = torch.topk(sims, k=k_search, dim=1)
        vals_np = vals.numpy()
        idx_np = idx.numpy()
        for row, (v_row, i_row) in enumerate(zip(vals_np, idx_np)):
            if exclude_self:
                global_row = start + row
                keep = i_row != global_row
                i_row = i_row[keep][:k]
                v_row = v_row[keep][:k]
            if len(i_row) < k:
                pad = k - len(i_row)
                i_row = np.pad(i_row, (0, pad), mode='edge')
                v_row = np.pad(v_row, (0, pad), mode='edge')
            scaled = v_row / max(temp, 1e-6)
            scaled = scaled - np.max(scaled)
            w = np.exp(scaled)
            w = w / np.sum(w)
            all_ids.append(value_ids[i_row])
            all_weights.append(w.astype(np.float32))
    return np.asarray(all_ids, dtype=np.int64), np.asarray(all_weights, dtype=np.float32)

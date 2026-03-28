from random import shuffle,choice
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import os


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.model(x)


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def mse_loss(real_item_emb, item_content_emb):
    loss = F.mse_loss(real_item_emb, item_content_emb)
    return loss


def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg


def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


#this version is from recbole
def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)


def next_batch_pairwise(data, batch_size, n_negs=1):
    training_data = data.training_data
    np.random.shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(data.item.keys())
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        total_num = len(users)
        u_idx = np.array([data.user[x] for x in users])
        value_ids = np.zeros(total_num, dtype=type(item_list[0]))
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list] = np.random.choice(item_list, size=len(check_list))
            check_list = np.array(
                [
                    i
                    for i, used, v in zip(
                        check_list,
                        data.training_set_uid[u_idx[check_list]],
                        value_ids[check_list]
                    )
                    if v in used
                ]
            )
        u_idx = u_idx.tolist()
        i_idx = [data.item[x] for x in items]
        j_idx = [data.item[x] for x in value_ids]
        yield u_idx, i_idx, j_idx


def next_batch_pairwise_LARA(data,batch_size,n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, u_neg_idx, i_neg_idx = [], [], [], []
        user_list = list(data.user.keys())
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            for m in range(n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                i_neg_idx.append(data.item[neg_item])
                neg_user = choice(user_list)
                while neg_user in data.training_set_i[items[i]]:
                    neg_user = choice(user_list)
                u_neg_idx.append(data.user[neg_user])
        yield u_idx, i_idx, u_neg_idx, i_neg_idx


def next_batch_pairwise_CLCRec(data, batch_size, n_negs=1):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    # Match official CLCRec (weiyinwei/CLCRec): negative pool is all items minus cold_set.
    cold_mapped = frozenset(data.mapped_cold_item_idx)
    item_list = [k for k in data.item.keys() if data.item[k] not in cold_mapped]
    if not item_list:
        raise ValueError(
            'next_batch_pairwise_CLCRec: warm-item negative pool is empty; check cold_item split.'
        )
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx = [], []
        for i, user in enumerate(users):
            u_idx.append([data.user[user]]*(1+n_negs))
            i_idx.append([data.item[items[i]]])
            # Without replacement (same as official CLCRec random.sample on pool \ user positives).
            candidates = [k for k in item_list if k not in data.training_set_u[user]]
            if len(candidates) < n_negs:
                raise ValueError(
                    f'next_batch_pairwise_CLCRec: user has only {len(candidates)} warm negatives '
                    f'available but n_negs={n_negs}.'
                )
            for neg_item in random.sample(candidates, n_negs):
                i_idx[i].append(data.item[neg_item])
        # u_idx [bs, 1+num_neg]
        # i_idx [bs, 1+num_neg]
        yield u_idx, i_idx



def next_batch_pairwise_CCFCRec(data, batch_size, positive_number, negative_number, self_neg_number):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    user_list = list(data.user.keys())
    item_list = list(data.item.keys())
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, neg_u_idx, pos_i_list, neg_i_list, self_neg_list = [], [], [], [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_user = choice(user_list)
            while neg_user in data.training_set_i[items[i]]:
                neg_user = choice(user_list)
            neg_u_idx.append(data.user[neg_user])
            positive_items_ = data.training_set_u[user]
            pos_i_list.append(list(np.random.choice(list(positive_items_), positive_number, replace=True)))
            tmp_neg_i_list = []
            for m in range(positive_number*negative_number):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                tmp_neg_i_list.append(neg_item)
            for mm in range(positive_number):
                start_idx = negative_number * mm
                end_idx = negative_number * (mm + 1)
                if mm == 0:
                    neg_i_list.append([tmp_neg_i_list[start_idx:end_idx]])
                else:
                    neg_i_list[i].append(tmp_neg_i_list[start_idx:end_idx])
            for m in range(self_neg_number):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                if m == 0:
                    self_neg_list.append([data.item[neg_item]])
                else:
                    self_neg_list[i].append(data.item[neg_item])
        yield u_idx, i_idx, neg_u_idx, pos_i_list, neg_i_list, self_neg_list


def next_batch_cgrc(data, batch_size, ranking_neg_per_user=32):
    """
    Batches for CGRC rating ranking loss (Eq. 8): one positive (u, i+) per row plus extra negatives
    so each minibatch has a shared item set B (positives ∪ sampled non-interacted items).
    """
    training_data = data.training_data
    np.random.shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_keys = list(data.item.keys())
    if not item_keys:
        raise ValueError('next_batch_cgrc: empty item set')
    while ptr < data_size:
        batch_end = min(ptr + batch_size, data_size)
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx = [data.user[u] for u in users]
        i_idx = [data.item[i] for i in items]
        B_set = set(i_idx)
        for u in users:
            uid = data.user[u]
            rated = data.training_set_uid[uid]
            added = 0
            tries = 0
            max_tries = ranking_neg_per_user * 50
            while added < ranking_neg_per_user and tries < max_tries:
                tries += 1
                j_key = np.random.choice(item_keys)
                if j_key not in rated:
                    B_set.add(data.item[j_key])
                    added += 1
        B_list = list(B_set)
        yield u_idx, i_idx, B_list


def set_seed(seed, cuda):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def process_bar(num, total):
    rate = float(num) / total
    ratenum = int(50 * rate)
    r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
    import sys
    sys.stdout.write(r)
    sys.stdout.flush()


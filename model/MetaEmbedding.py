import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise
import torch.nn.functional as F


# Following the source code process: https://github.com/hsuchengmath/MetaEmbedding_torch
# Here the pretrained warm embedding is adopted as the target of support set, and thus the whole train set is utilized as the target of query set
class MetaEmbedding(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(MetaEmbedding, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)

        self.model = MetaEmbedding_Learner(args, self.data, self.emb_size, device, self.args.lr/10.)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_i_idx, neg_i_idx = batch
                user_idx = torch.LongTensor(user_idx).to(self.device)
                pos_i_idx = torch.LongTensor(pos_i_idx).to(self.device)
                neg_i_idx = torch.LongTensor(neg_i_idx).to(self.device)
                batch_user_idxs = torch.cat([user_idx, user_idx], dim=0)
                batch_item_idxs = torch.cat([pos_i_idx, neg_i_idx], dim=0)
                # first term
                y_hat_first = self.model(batch_user_idxs, batch_item_idxs, meta_ID_emb=None, warm_or_cold='cold')
                meta_ID_emb = self.model.meta_ID_emb
                batch_targets = torch.cat([torch.ones_like(pos_i_idx, dtype=torch.float32),
                                           torch.zeros_like(neg_i_idx, dtype=torch.float32)], dim=0)
                # calculate loss (1)
                batch_loss_a = torch.mean(F.binary_cross_entropy_with_logits(y_hat_first, batch_targets))
                # second term
                y_hat = self.model(batch_user_idxs, batch_item_idxs, cold_loss_a=batch_loss_a, meta_ID_emb=meta_ID_emb,
                                   warm_or_cold='cold')
                # calculate loss (2)
                #batch_y_b_tensor = torch.tensor(batch_y_b).view(-1, 1).type(torch.FloatTensor)
                batch_loss_b = torch.mean(F.binary_cross_entropy_with_logits(y_hat, batch_targets))
                # opt loss
                ME_loss = batch_loss_a * self.args.alpha + batch_loss_b * (1 - self.args.alpha)
                optimizer.zero_grad()
                ME_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', ME_loss.item())

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = self.model.get_all_embs()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if self.args.cold_object == 'item':
                    now_cold_item_gen_emb = self.model.get_cold_embs(self.data.mapped_cold_item_idx)
                    self.item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_gen_emb
                else:
                    now_cold_user_gen_emb = self.model.get_cold_embs(self.data.mapped_cold_user_idx)
                    self.user_emb.data[self.data.mapped_cold_user_idx] = now_cold_user_gen_emb
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
            now_best_user_emb, now_best_item_emb = self.model.get_all_embs()
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            if self.args.cold_object == 'item':
                now_cold_item_gen_emb = self.model.get_cold_embs(self.data.mapped_cold_item_idx)
                self.best_item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_gen_emb
            else:
                now_cold_user_gen_emb = self.model.get_cold_embs(self.data.mapped_cold_user_idx)
                self.best_user_emb.data[self.data.mapped_cold_user_idx] = now_cold_user_gen_emb

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


class MetaEmbedding_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device, cold_lr):
        super(MetaEmbedding_Learner, self).__init__()
        self.args = args
        self.emb_size = emb_size
        self.data = data
        self.device = device
        self.cold_lr = cold_lr
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self.emb_pred_Dense = nn.Linear(self.content_dim, self.emb_size)
        self.embedding_dict = self._init_model()

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt',
                                   map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt',
                                   map_location='cpu'),
        })
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict

    def generate_meta_emb(self, ui_contents):
        """
        This is the simplest architecture of the embedding generator,
        with only a dense layer.
        You can customize it if you want have a stronger performance,
        for example, you can add an l2 regularization term or alter
        the pooling layer.
        """
        emb_pred = self.emb_pred_Dense(ui_contents) / 5.
        return emb_pred

    def get_embeddings_and_info(self, batch_u, batch_i):
        user_emb = self.embedding_dict['user_emb'][batch_u]
        item_embs = self.embedding_dict['item_emb'][batch_i]
        if self.args.cold_object == 'item':
            item_contents = self.item_content[batch_i]
            return user_emb, item_embs, item_contents
        else:
            user_contents = self.user_content[batch_u]
            return user_emb, item_embs, user_contents

    def get_all_embs(self):
        user_emb = self.embedding_dict['user_emb']
        item_embs = self.embedding_dict['item_emb']
        return user_emb, item_embs

    def get_cold_embs(self, cold_idxs):
        if self.args.cold_object == 'item':
            item_cont = self.item_content[cold_idxs]
            meta_ID_emb = self.generate_meta_emb(item_cont)
        else:
            user_cont = self.user_content[cold_idxs]
            meta_ID_emb = self.generate_meta_emb(user_cont)
        return meta_ID_emb

    def get_yhat(self, user_emb, item_embs):
        y_hat = torch.sum(user_emb * item_embs, 1)
        return y_hat

    def forward(self, batch_u, batch_i, cold_loss_a=None, meta_ID_emb=None, warm_or_cold=str):
        # get lookup embedding
        user_embs, item_embs, ui_contents = self.get_embeddings_and_info(batch_u, batch_i)
        # main model
        if warm_or_cold == 'warm':
            y_hat = self.get_yhat(user_embs, item_embs)
            return y_hat
        elif warm_or_cold == 'cold':
            # Meta-Embedding: step 1, cold-start,
            #     use the generated meta-embedding to make predictions
            #     and calculate the cold-start loss_a
            if meta_ID_emb is None:
                if self.args.cold_object == 'item':
                    meta_ID_emb = self.generate_meta_emb(ui_contents)
                    self.meta_ID_emb = meta_ID_emb
                    cold_yhat_a = self.get_yhat(user_embs, meta_ID_emb)
                    return cold_yhat_a
                else:
                    meta_ID_emb = self.generate_meta_emb(ui_contents)
                    self.meta_ID_emb = meta_ID_emb
                    cold_yhat_a = self.get_yhat(meta_ID_emb, item_embs)
                    return cold_yhat_a
            else:
                # Meta-Embedding: step 2, apply gradient descent once
                #     get the adapted embedding
                # cold_emb_grads = tf.gradients(cold_loss_a, meta_ID_emb)[0]
                cold_emb_grads = torch.autograd.grad(cold_loss_a, meta_ID_emb, retain_graph=True)[0]
                meta_ID_emb_new = meta_ID_emb - self.cold_lr * cold_emb_grads
                # Meta-Embedding: step 3,
                #     use the adapted embedding to make prediction on another mini-batch
                #     and calculate the warm-up loss_b
                if self.args.cold_object == 'item':
                    cold_yhat_b = self.get_yhat(user_embs, meta_ID_emb_new)
                else:
                    cold_yhat_b = self.get_yhat(meta_ID_emb_new, item_embs)
                return cold_yhat_b

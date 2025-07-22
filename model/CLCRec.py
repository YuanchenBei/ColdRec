import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise_CLCRec
import torch.nn.functional as F


# Following the source code process: https://github.com/weiyinwei/CLCRec
class CLCRec(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(CLCRec, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                   warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                   warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                   user_content=user_content, item_content=item_content)
        if self.args.cold_object == 'user':
            raise Exception('Cold user is not supported in CLCRec due to its specific design for item cold-start problem.')
        self.model = CLCRec_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise_CLCRec(self.data, self.batch_size, self.args.num_neg)):
                user_idxs, item_idxs = batch
                user_idxs = torch.LongTensor(user_idxs).to(self.device)
                item_idxs = torch.LongTensor(item_idxs).to(self.device)
                batch_loss = model.loss(user_idxs, item_idxs)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb, now_cold_item_emb = self.model.get_all_embs()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                self.item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_emb
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
            now_best_user_emb, now_best_item_emb, now_best_cold_item_emb = self.model.get_all_embs()
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            self.best_item_emb.data[self.data.mapped_cold_item_idx] = now_best_cold_item_emb

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


class CLCRec_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(CLCRec_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        self.MLP = nn.Linear(emb_size, emb_size)
        self.encoder_layer1 = nn.Linear(self.content_dim, 256)
        self.encoder_layer2 = nn.Linear(256, emb_size)
        self.att_weight_1 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((emb_size, emb_size))))
        self.att_weight_2 = nn.Parameter(nn.init.kaiming_normal_(torch.rand((emb_size, emb_size))))
        self.bias = nn.Parameter(nn.init.kaiming_normal_(torch.rand((emb_size, 1))))
        self.att_sum_layer = nn.Linear(emb_size, emb_size)
        self.num_sample = 0.5
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def encoder(self):
        if self.args.cold_object == 'item':
            feature = self.item_content
        else:
            feature = self.user_content
        feature = F.leaky_relu(self.encoder_layer1(feature))
        feature = self.encoder_layer2(feature)
        return feature

    def loss_contrastive(self, tensor_anchor, tensor_all, temp_value):
        all_score = torch.exp(torch.sum(tensor_anchor * tensor_all, dim=1) / temp_value).view(-1, 1 + self.args.num_neg)
        all_score = all_score.view(-1, 1 + self.args.num_neg)
        pos_score = all_score[:, 0]
        all_score = torch.sum(all_score, dim=1)
        contrastive_loss = (-torch.log(pos_score / all_score)).mean()
        return contrastive_loss

    def forward(self, user_tensor, item_tensor):
        pos_item_tensor = item_tensor[:, 0].unsqueeze(1)
        pos_item_tensor = pos_item_tensor.repeat(1, 1 + self.args.num_neg).view(-1, 1).squeeze()

        user_tensor = user_tensor.view(-1, 1).squeeze()
        item_tensor = item_tensor.view(-1, 1).squeeze()

        feature = self.encoder()
        all_item_feat = feature[item_tensor]

        user_embedding = self.embedding_dict['user_emb'][user_tensor]
        pos_item_embedding = self.embedding_dict['item_emb'][pos_item_tensor]
        all_item_embedding = self.embedding_dict['item_emb'][item_tensor]

        head_feat = F.normalize(all_item_feat, dim=1)
        head_embed = F.normalize(pos_item_embedding, dim=1)

        all_item_input = all_item_embedding.clone()
        rand_index = torch.randint(all_item_embedding.size(0), (int(all_item_embedding.size(0) * self.num_sample),)).to(self.device)
        all_item_input[rand_index] = all_item_feat[rand_index].clone()

        self.contrastive_loss_1 = self.loss_contrastive(head_embed, head_feat, self.args.temp_value)
        self.contrastive_loss_2 = self.loss_contrastive(user_embedding, all_item_input, self.args.temp_value)
        reg_loss = ((torch.sqrt((user_embedding ** 2).sum(1))).mean() + (torch.sqrt((all_item_embedding ** 2).sum(1))).mean()) / 2
        return self.contrastive_loss_1 * self.args.lr_lambda + (self.contrastive_loss_2) * (1 - self.args.lr_lambda), reg_loss

    def loss(self, user_tensor, item_tensor):
        contrastive_loss, reg_loss = self.forward(user_tensor, item_tensor)
        reg_loss = self.args.reg * reg_loss
        return reg_loss + contrastive_loss

    def get_all_embs(self):
        feature = self.encoder()
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb'], feature[self.data.mapped_cold_item_idx]

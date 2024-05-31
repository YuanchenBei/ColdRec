import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise_CCFCRec


# Following the source code process: https://github.com/zzhin/CCFCRec
class CCFCRec(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content, item_content):
        super(CCFCRec, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                   warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                   warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                   user_content=user_content, item_content=item_content)
        if self.args.cold_object == 'user':
            raise Exception('Cold user is not supported in CCFCRec due to its specific design for item cold-start problem.')
        self.model = CCFCRec_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise_CCFCRec(self.data, self.batch_size, self.args.positive_number, self.args.negative_number, self.args.self_neg_number)):
                u_idx, i_idx, neg_u_idx, pos_i_list, neg_i_list, self_neg_list = batch
                u_idx = torch.LongTensor(u_idx).to(self.device)
                i_idx = torch.LongTensor(i_idx).to(self.device)
                neg_u_idx = torch.LongTensor(neg_u_idx).to(self.device)
                pos_i_list = torch.LongTensor(pos_i_list).to(self.device)
                neg_i_list = torch.LongTensor(neg_i_list).to(self.device)
                self_neg_list = torch.LongTensor(self_neg_list).to(self.device)
                # run model
                q_v_c = model(u_idx, i_idx)
                q_v_c_unsqueeze = q_v_c.unsqueeze(dim=1)
                positive_item_emb = model.item_embedding[pos_i_list]
                pos_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, positive_item_emb), dim=2) / (self.args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(positive_item_emb, dim=2))
                pos_contrast_exp = torch.exp(pos_contrast_mul)
                neg_item_emb = model.item_embedding[neg_i_list]
                q_v_c_un2squeeze = q_v_c_unsqueeze.unsqueeze(dim=1)
                neg_contrast_mul = torch.sum(torch.mul(q_v_c_un2squeeze, neg_item_emb), dim=3) / (self.args.tau * torch.norm(q_v_c_un2squeeze, dim=3) * torch.norm(neg_item_emb, dim=3))
                neg_contrast_exp = torch.exp(neg_contrast_mul)
                neg_contrast_sum = torch.sum(neg_contrast_exp, dim=2)
                contrast_val = -torch.log(pos_contrast_exp / (pos_contrast_exp + neg_contrast_sum))
                contrast_sum = torch.sum(torch.sum(contrast_val, dim=1), dim=0) / contrast_val.shape[1]
                '''
                contrast self
                '''
                self_neg_item_emb = model.item_embedding[self_neg_list]
                self_neg_contrast_mul = torch.sum(torch.mul(q_v_c_unsqueeze, self_neg_item_emb), dim=2) / (self.args.tau * torch.norm(q_v_c_unsqueeze, dim=2) * torch.norm(self_neg_item_emb, dim=2))
                self_neg_contrast_sum = torch.sum(torch.exp(self_neg_contrast_mul), dim=1)
                item_emb = model.item_embedding[i_idx]
                self_pos_contrast_mul = torch.sum(torch.mul(q_v_c, item_emb), dim=1) / (self.args.tau * torch.norm(q_v_c, dim=1) * torch.norm(item_emb, dim=1))
                self_pos_contrast_exp = torch.exp(self_pos_contrast_mul)  # shape = 1024*1
                self_contrast_val = -torch.log(self_pos_contrast_exp / (self_pos_contrast_exp + self_neg_contrast_sum))
                self_contrast_sum = torch.sum(self_contrast_val)
                # rank loss
                user_emb = model.user_embedding[u_idx]
                item_emb = model.item_embedding[i_idx]
                neg_user_emb = model.user_embedding[neg_u_idx]
                logsigmoid = torch.nn.LogSigmoid()
                y_uv = torch.mul(item_emb, user_emb).sum(dim=1)
                y_kv = torch.mul(item_emb, neg_user_emb).sum(dim=1)
                y_ukv = -logsigmoid(y_uv - y_kv).sum()
                y_uv2 = torch.mul(q_v_c, user_emb).sum(dim=1)
                y_kv2 = torch.mul(q_v_c, neg_user_emb).sum(dim=1)
                y_ukv2 = -logsigmoid(y_uv2 - y_kv2).sum()
                batch_loss = self.args.lambda1 * (contrast_sum + self_contrast_sum) + (1 - self.args.lambda1) * (y_ukv + y_ukv2)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                now_user_emb = self.model.user_embedding
                now_item_emb = self.model.item_embedding
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')

        model.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(self.user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def save(self):
        with torch.no_grad():
            now_best_user_emb = self.model.user_embedding
            now_best_item_emb = self.model.item_embedding
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class CCFCRec_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(CCFCRec_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        self.attr_matrix = torch.nn.Parameter(torch.FloatTensor(self.data.item_content_dim, self.args.attr_present_dim))
        self.attr_W1 = torch.nn.Parameter(torch.FloatTensor(self.args.attr_present_dim, self.args.attr_present_dim))
        self.attr_b1 = torch.nn.Parameter(torch.FloatTensor(self.args.attr_present_dim, 1))
        self.attr_W2 = torch.nn.Parameter(torch.FloatTensor(self.args.attr_present_dim, 1))
        self.h = nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        if self.args.pretrain is True:
            if self.args.pretrain_update is True:
                self.user_embedding = nn.Parameter(torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'), requires_grad=True)
                self.item_embedding = nn.Parameter(torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'), requires_grad=True)
            else:
                self.user_embedding = nn.Parameter(torch.load('user_emb.pt'), requires_grad=False)
                self.item_embedding = nn.Parameter(torch.load('item_emb.pt'), requires_grad=False)
        else:
            self.user_embedding = nn.Parameter(torch.FloatTensor(self.data.user_num, self.args.implicit_dim))
            self.item_embedding = nn.Parameter(torch.FloatTensor(self.data.item_num, self.args.implicit_dim))
        self.gen_layer1 = nn.Linear(self.args.attr_present_dim, self.args.cat_implicit_dim)
        self.gen_layer2 = nn.Linear(self.args.attr_present_dim, self.args.attr_present_dim)
        self.__init_param__()

    def __init_param__(self):
        nn.init.xavier_normal_(self.attr_matrix)
        nn.init.xavier_normal_(self.attr_W1)
        nn.init.xavier_normal_(self.attr_W2)
        nn.init.xavier_normal_(self.attr_b1)
        if self.args.pretrain is False:
            nn.init.xavier_normal_(self.user_embedding)
            nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.gen_layer1.weight)
        nn.init.xavier_normal_(self.gen_layer2.weight)

    def forward(self, u_idx, i_idx):
        attribute = self.item_content[i_idx]
        batch_size = u_idx.shape[0]
        z_v = torch.matmul(torch.matmul(self.attr_matrix, self.attr_W1) + self.attr_b1.squeeze(), self.attr_W2)
        z_v_copy = z_v.repeat(batch_size, 1, 1)
        z_v_squeeze = z_v_copy.squeeze(dim=2).to(self.device)
        neg_inf = torch.full(z_v_squeeze.shape, -1e6).to(self.device)
        z_v_mask = torch.where(attribute != -1, z_v_squeeze, neg_inf)
        attr_attention_weight = torch.softmax(z_v_mask, dim=1)
        final_attr_emb = torch.matmul(attr_attention_weight, self.attr_matrix)
        q_v_a = final_attr_emb
        q_v_c = self.gen_layer2(self.h(self.gen_layer1(q_v_a)))
        return q_v_c
import torch
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss
import faiss
from .MF import Matrix_Factorization
from .LightGCN import LGCN_Encoder


class KNN(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(KNN, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)
        if self.args.backbone == 'MF':
            self.encoder = Matrix_Factorization(self.data, self.emb_size)
        else:
            self.encoder = LGCN_Encoder(self.data, self.emb_size, self.args.layers, self.device)
        self.knn_num = args.knn_num

    def train(self):
        encoder = self.encoder.to(self.device)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.lr)
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            encoder.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = encoder()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                encoder.eval()
                now_user_emb, now_item_emb = self.encoder()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if self.args.cold_object == 'item':
                    cold_item_content = self.data.mapped_item_content[self.data.mapped_cold_item_idx]
                    warm_item_content = self.data.mapped_item_content[self.data.mapped_warm_item_idx]
                    cold_generated_emb = self.knn_search(cold_item_content, warm_item_content, now_item_emb)
                    self.item_emb.data[self.data.mapped_cold_item_idx] = cold_generated_emb
                else:
                    cold_user_content = self.data.mapped_user_content[self.data.mapped_cold_user_idx]
                    warm_user_content = self.data.mapped_user_content[self.data.mapped_warm_user_idx]
                    cold_generated_emb = self.knn_search(cold_user_content, warm_user_content, now_user_emb)
                    self.user_emb.data[self.data.mapped_cold_user_idx] = cold_generated_emb
                if epoch % 5 == 0:
                    self.fast_evaluation(epoch, valid_type='all')
                    if self.early_stop_flag:
                        if self.early_stop_patience <= 0:
                            break

        self.timer(start=False)
        encoder.eval()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        if self.args.save_emb:
            torch.save(self.user_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_user_emb.pt")
            torch.save(self.item_emb, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_item_emb.pt")

    def knn_search(self, content_query, content_value, emb_table):
        if self.args.cold_object == 'item':
            self.model = faiss.IndexFlatIP(self.data.item_content_dim)
        else:
            self.model = faiss.IndexFlatIP(self.data.user_content_dim)
        self.model.add(content_value)
        D, I = self.model.search(content_query, self.knn_num)
        if self.args.cold_object == 'item':
            mapped_ids = self.data.mapped_warm_item_idx[I]
        else:
            mapped_ids = self.data.mapped_warm_user_idx[I]
        mapped_ids = torch.LongTensor(mapped_ids).to(self.device)
        cold_generated_emb = torch.mean(emb_table[mapped_ids], dim=1)
        return cold_generated_emb

    def save(self):
        with torch.no_grad():
            now_best_user_emb, now_best_item_emb = self.encoder.forward()
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()
            if self.args.cold_object == 'item':
                cold_item_content = self.data.mapped_item_content[self.data.mapped_cold_item_idx]
                warm_item_content = self.data.mapped_item_content[self.data.mapped_warm_item_idx]
                cold_generated_emb = self.knn_search(cold_item_content, warm_item_content, now_best_item_emb)
                self.best_item_emb.data[self.data.mapped_cold_item_idx] = cold_generated_emb
            else:
                cold_user_content = self.data.mapped_user_content[self.data.mapped_cold_user_idx]
                warm_user_content = self.data.mapped_user_content[self.data.mapped_warm_user_idx]
                cold_generated_emb = self.knn_search(cold_user_content, warm_user_content, now_best_user_emb)
                self.best_user_emb.data[self.data.mapped_cold_user_idx] = cold_generated_emb

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


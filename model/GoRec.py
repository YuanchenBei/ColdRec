import torch
import torch.nn as nn
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise
from sklearn.cluster import KMeans
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# Following the source code process: https://github.com/HaoyueBai98/GoRec
class GoRec(BaseColdStartTrainer):
    def __init__(self, args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                 warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                 warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                 user_content=None, item_content=None):
        super(GoRec, self).__init__(args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
                                  warm_test_data, cold_test_data, all_test_data, user_num, item_num,
                                  warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device,
                                  user_content=user_content, item_content=item_content)

        self.model = GoRec_Learner(args, self.data, self.emb_size, device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion_mse = torch.nn.MSELoss()
        criterion_uni = uniformity
        criterion_kl = torch.nn.KLDivLoss()
        model.clustering()
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                u_idx, i_idx, _ = batch
                if self.args.cold_object == 'item':
                    warm = self.model.embedding_dict['item_emb'][i_idx]
                    side_information = self.model.item_content[i_idx]
                else:
                    warm = self.model.embedding_dict['user_emb'][u_idx]
                    side_information = self.model.user_content[u_idx]
                side_information = torch.nn.functional.normalize(side_information)
                rec_warm, mu, log_variances, z, zgc = self.model(warm, side_information)
                rec_loss = criterion_mse(rec_warm, warm.to(self.device))
                uni_loss = criterion_uni(mu)
                uni_loss = self.args.uni_coeff * uni_loss
                z = F.softmax(z, dim=1)
                zgc = F.softmax(zgc, dim=1)
                log_z = torch.log(z + 1e-10)
                kl_loss = criterion_kl(log_z, zgc)
                kl_loss = self.args.kl_coeff * kl_loss
                batch_loss = rec_loss + uni_loss +kl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                model.eval()
                if self.args.cold_object == 'item':
                    now_user_emb = self.model.embedding_dict['user_emb']
                    now_item_emb = self.model(self.model.embedding_dict['item_emb'], self.model.item_content)
                else:
                    now_user_emb = self.model(self.model.embedding_dict['user_emb'], self.model.user_content)
                    now_item_emb = self.model.embedding_dict['item_emb']
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                if epoch % 5 == 0:
                    model.eval()
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
            torch.save(self.model.cluster_label, f"./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.model}_cluster_label.pt")

    def save(self):
        with torch.no_grad():
            if self.args.cold_object == 'item':
                now_best_user_emb = self.model.embedding_dict['user_emb']
                now_best_item_emb = self.model(self.model.embedding_dict['item_emb'], self.model.item_content)
            else:
                now_best_user_emb = self.model(self.model.embedding_dict['user_emb'], self.model.user_content)
                now_best_item_emb = self.model.embedding_dict['item_emb']
            self.best_user_emb = now_best_user_emb.clone()
            self.best_item_emb = now_best_item_emb.clone()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class GoRec_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device, decoder_layer=2):
        super(GoRec_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)
        self.z_size = emb_size
        self.encoder = Encoder(latent_dim=self.latent_size, z_size=self.z_size, si_dim=self.content_dim)
        self.decoder = Decoder(z_size=self.z_size, latent_dim=self.latent_size, layer=decoder_layer, si_dim=self.content_dim)
        self.latent = emb_size
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.embedding_dict = self._init_model()

    def _init_model(self):
        embedding_dict = nn.ParameterDict({
            'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
            'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
        })
        embedding_dict['user_emb'].requires_grad = False
        embedding_dict['item_emb'].requires_grad = False
        return embedding_dict

    def clustering(self):
        cluster_fit = KMeans(n_clusters=self.args.pre_cluster_num, random_state=0).fit(self.embedding_dict[self.args.cold_object+'_emb'].cpu().detach().numpy())
        self.cluster_label = cluster_fit.labels_
        cluster_cfmean = []
        for i in range(self.args.pre_cluster_num):
            cluster_cfmean.append(self.embedding_dict[self.args.cold_object+'_emb'][np.where(self.cluster_label == i)].mean(0))
        cluster_cfmean = torch.stack(cluster_cfmean).to(self.device)
        self.cluster_cfmean = cluster_cfmean[self.cluster_label]
        return self.cluster_cfmean

    def forward(self, warm, side_information, gen_size=10):
        if self.training:
            original = warm

            # encode
            mu, log_variances, mu_zgc, log_variances_zgc = self.encoder(warm, side_information)

            # we need true variance not log
            variances = torch.exp(log_variances * 0.5)
            variances_zgc = torch.exp(log_variances_zgc * 0.5)

            # sample from gaussian
            sample_from_normal = Variable(torch.randn(len(warm), self.z_size).to(self.device), requires_grad=True)
            sample_from_normal_zgc = Variable(torch.randn(len(warm), self.z_size).to(self.device), requires_grad=True)

            # shift and scale using mean and variances
            z = sample_from_normal * variances + mu
            zgc = sample_from_normal_zgc * variances_zgc + mu_zgc

            # decode tensor
            side_information = self.dropout(side_information)
            rec_warm = self.decoder(z, side_information)

            return rec_warm, mu, log_variances, z, zgc
        else:
            if warm is None:
                # just sample and decode
                z = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)

            else:
                mu, log_variances, _, _ = self.encoder(warm, side_information)

                # _, _, mu, log_variances = self.encoder(warm, side_information)
                # we need true variance not log
                variances = torch.exp(log_variances * 0.5)

                # sample from gaussian
                sample_from_normal = Variable(torch.randn(len(warm), self.z_size).to(self.device),
                                              requires_grad=True)

                # shift and scale using mean and variances
                # z = sample_from_normal * variances + mu
                z = mu

            # decode tensor
            rec_warm = self.decoder(z, side_information)
            return rec_warm


def l2_regularization(model):
    l2_loss = []
    for name, parameters in model.named_parameters():
        l2_loss.append((parameters ** 2).sum() / 2.0)
    return sum(l2_loss)


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input, out=False):
        output = self.linear(input)
        return output


class Encoder(nn.Module):
    def __init__(self, latent_dim, z_size, si_dim):
        super(Encoder, self).__init__()
        self.size = latent_dim

        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + si_dim), out_features=(latent_dim), bias=False),
                                nn.BatchNorm1d(num_features=latent_dim),
                                nn.Tanh())
        self.l_mu = nn.Linear(in_features=self.size, out_features=z_size)
        self.l_var = nn.Linear(in_features=self.size, out_features=z_size)

        self.l_mu_zgc = nn.Linear(in_features=si_dim, out_features=z_size)
        self.l_var_zgc = nn.Linear(in_features=si_dim, out_features=z_size)

        nn.init.xavier_uniform_(self.l_mu.weight)
        nn.init.xavier_uniform_(self.l_var.weight)
        nn.init.xavier_uniform_(self.l_mu_zgc.weight)
        nn.init.xavier_uniform_(self.l_var_zgc.weight)

    def forward(self, warm, side_information):
        # warm = self.inference(warm)
        #side_information_z = self.content_fc(side_information)
        mu_zgc = self.l_mu_zgc(side_information)
        logvar_zgc = self.l_var_zgc(side_information)

        warm = torch.cat((side_information, warm), 1)
        warm = self.fc(warm)
        mu = self.l_mu(warm)
        logvar = self.l_var(warm)
        return mu, logvar, mu_zgc, logvar_zgc


class Decoder(nn.Module):
    def __init__(self, z_size, latent_dim, layer, si_dim):
        super(Decoder, self).__init__()
        # start from B * z_size
        # concatenate one hot encoded class vector
        self.fc = nn.Sequential(nn.Linear(in_features=(z_size + si_dim), out_features=(latent_dim), bias=False),
                                nn.BatchNorm1d(num_features=latent_dim),
                                nn.Tanh())
        self.size = latent_dim
        layers = []
        for i in range(layer):
            layers.append(EncoderBlock(input_dim=self.size, output_dim=64))
            self.size = latent_dim

        self.geneator = nn.Sequential(*layers)

    def forward(self, z, side_information):
        z_cat = torch.cat((side_information, z), 1)
        rec_warm = self.fc(z_cat)
        rec_warm = self.geneator(rec_warm)
        return rec_warm

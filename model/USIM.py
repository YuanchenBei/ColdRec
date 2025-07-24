import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseRecommender import BaseColdStartTrainer
from util.utils import next_batch_pairwise, bpr_loss, l2_reg_loss

# Modified from the original code: https://github.com/Ruochen1003/USIM
class USIM(BaseColdStartTrainer):
    """USIM Cold-Start Recommendation Model - Item Mapping based on Reinforcement Learning"""
    def __init__(self, config):
        super(USIM, self).__init__(config)
        if self.args.cold_object == 'user':
            raise Exception('USIM currently only supports item cold-start, user cold-start is not supported yet')
        self.model = USIM_Learner(self.args, self.data, self.emb_size, self.device)

    def train(self):
        model = self.model.to(self.device)
        rec_optimizer = torch.optim.Adam(model.embedding_dict.parameters(), lr=self.lr)
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=getattr(self.args, 'actor_lr', 0.001))
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=getattr(self.args, 'critic_lr', 0.001))
        self.timer(start=True)
        for epoch in range(self.maxEpoch):
            model.train()
            # Recommendation part
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_emb, pos_item_emb, neg_item_emb = model.get_training_embs(user_idx, pos_idx, neg_idx)
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                rec_optimizer.zero_grad()
                (rec_loss + reg_loss).backward()
                rec_optimizer.step()
                # Store detached embeddings in buffer
                model.add_experience_to_buffer(user_emb.detach(), pos_item_emb.detach(), neg_item_emb.detach())
                if n % 50 == 0:
                    print(f'training: {epoch + 1} batch {n} rec_loss: {rec_loss.item():.4f}')
            # RL part
            if len(model.buffer) > model.batch_size_rl:
                for _ in range(5):
                    rl_loss = model.compute_rl_loss()
                    if rl_loss > 0:
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        rl_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                        model._update_target_networks()
            # Evaluation
            with torch.no_grad():
                model.eval()
                now_user_emb, now_item_emb = model.get_all_embs()
                self.user_emb = now_user_emb.clone()
                self.item_emb = now_item_emb.clone()
                cold_item_emb = model.generate_cold_item_emb()
                self.item_emb.data[self.data.mapped_cold_item_idx] = cold_item_emb
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
            now_cold_item_emb = self.model.generate_cold_item_emb()
            self.best_item_emb.data[self.data.mapped_cold_item_idx] = now_cold_item_emb

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

class USIM_Learner(nn.Module):
    def __init__(self, args, data, emb_size, device):
        super(USIM_Learner, self).__init__()
        self.args = args
        self.latent_size = emb_size
        self.device = device
        self.data = data
        self.embedding_dict = self._load_pretrained_embeddings()
        self.content_dim = self.data.item_content_dim if self.args.cold_object == 'item' else self.data.user_content_dim
        self.content_mapping = ContentMapping(self.content_dim, self.latent_size, device)
        self.actor = Actor(self.latent_size, device)
        self.critic = Critic(self.latent_size, device)
        self.buffer_size = getattr(self.args, 'buffer_size', 10000)
        self.buffer = ReplayBuffer(self.buffer_size, self.latent_size, device)
        self.gamma = getattr(self.args, 'gamma', 0.99)
        self.tau = getattr(self.args, 'tau', 0.005)
        self.batch_size_rl = getattr(self.args, 'batch_size_rl', 32)
        self.target_actor = Actor(self.latent_size, device)
        self.target_critic = Critic(self.latent_size, device)
        self._update_target_networks(tau=1.0)
        if self.args.cold_object == 'item':
            self.item_content = torch.tensor(self.data.mapped_item_content, dtype=torch.float32, requires_grad=False).to(device)
        else:
            self.user_content = torch.tensor(self.data.mapped_user_content, dtype=torch.float32, requires_grad=False).to(device)

    def _load_pretrained_embeddings(self):
        try:
            embedding_dict = nn.ParameterDict({
                'user_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_user_emb.pt', map_location='cpu'),
                'item_emb': torch.load(f'./emb/{self.args.dataset}_cold_{self.args.cold_object}_{self.args.backbone}_item_emb.pt', map_location='cpu'),
            })
            print(f"Successfully loaded pre-trained {self.args.backbone} embeddings")
        except FileNotFoundError:
            print(f"Pre-trained {self.args.backbone} embeddings not found")
            initializer = nn.init.xavier_uniform_
            embedding_dict = nn.ParameterDict({
                'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
                'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            })
        return embedding_dict

    def get_training_embs(self, uid, iid, nid):
        user_emb = self.embedding_dict['user_emb'][uid]
        pos_item_emb = self.embedding_dict['item_emb'][iid]
        neg_item_emb = self.embedding_dict['item_emb'][nid]
        return user_emb, pos_item_emb, neg_item_emb

    def get_all_embs(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']

    def generate_cold_item_emb(self):
        if self.args.cold_object == 'item':
            cold_content = self.item_content[self.data.mapped_cold_item_idx]
            cold_emb = self.content_mapping(cold_content)
            return cold_emb
        else:
            cold_content = self.user_content[self.data.mapped_cold_user_idx]
            cold_emb = self.content_mapping(cold_content)
            return cold_emb

    def add_experience_to_buffer(self, user_emb, pos_item_emb, neg_item_emb):
        batch_size = user_emb.size(0)
        for i in range(batch_size):
            state = user_emb[i]
            action = pos_item_emb[i]
            reward = 1.0
            next_state = user_emb[i]
            self.buffer.add(state, action, reward, next_state)

    def compute_rl_loss(self):
        batch_size = self.buffer.size
        if batch_size < self.batch_size_rl:
            return torch.tensor(0.0, device=self.device)
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size_rl)
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        target_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, target_actions)
        target_q = rewards + self.gamma * target_q
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        return actor_loss + critic_loss

    def _update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class ContentMapping(nn.Module):
    def __init__(self, content_dim, latent_size, device):
        super(ContentMapping, self).__init__()
        self.device = device
        self.mapping = nn.Sequential(
            nn.Linear(content_dim, 2 * latent_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2 * latent_size, latent_size),
            nn.Tanh()
        )
    def forward(self, content):
        return self.mapping(content)

class Actor(nn.Module):
    def __init__(self, latent_size, device):
        super(Actor, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, latent_size),
            nn.Tanh()
        )
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self, latent_size, device):
        super(Critic, self).__init__()
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(2 * latent_size, 2 * latent_size),
            nn.ReLU(),
            nn.Linear(2 * latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity, latent_size, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        self.states = torch.zeros(capacity, latent_size, device=device)
        self.actions = torch.zeros(capacity, latent_size, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.next_states = torch.zeros(capacity, latent_size, device=device)
    def add(self, state, action, reward, next_state):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices]
        )
    def __len__(self):
        return self.size

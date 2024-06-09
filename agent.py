import torch
from torch import optim
import numpy as np
import random
from torch.nn import functional as F
from .models import PolicyNetworkFC, QNetworkFC
from torch.distributions.transforms import TanhTransform, AffineTransform, ComposeTransform
# from utils import ReplayBuffer
import pdb


class Agent():
    def __init__(self, state_dim, action_dim, alpha=.2, hidden_size=128,
                 batch_size=256, gamma=0.99, replay_size=400000, polyak=0.995,
                 tb_writer=None, train_alpha=False, action_limit=1., lr=1e-3,
                 device="cpu", target_entropy=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.device = device
        self.min_transitions = int(1e4)
        self.polyak = polyak
        # self.replay_buffer = ReplayBuffer(state_dim, action_dim, replay_size, self.device)
        self.tb_writer = tb_writer
        self.num_updates = 0
        log_alpha = torch.log(torch.zeros(1)+alpha)
        self._alpha_param = torch.tensor(log_alpha, requires_grad=train_alpha, device=device)
        if target_entropy is None:
            self.target_entropy = -torch.tensor([action_dim], device=device)
        else:
            self.target_entropy = torch.tensor([target_entropy], device=device,
                                               dtype=torch.float32)


        if train_alpha:
            self.alpha_optim = optim.Adam([self._alpha_param], lr=self.lr)
        else:
            self.alpha_optim = None

        self.action_limit = action_limit
        if action_limit is not None:
            tanh_tf = TanhTransform()
            aff_tf = AffineTransform(0, action_limit)
            self.action_transform = ComposeTransform([tanh_tf, aff_tf])
        else:
            # Placeholder transform
            self.action_transform = AffineTransform(0, 1)

        self.inv_action_transform = self.action_transform.inv

        # Policy network
        self.policy_net = PolicyNetworkFC(state_dim, action_dim, hidden_size).to(device)

        # Q networks
        self.q_net1 = QNetworkFC(state_dim, action_dim, hidden_size).to(device)
        self.q_net2 = QNetworkFC(state_dim, action_dim, hidden_size).to(device)

        # Target Q networks
        self.target_q_net1 = QNetworkFC(state_dim, action_dim, hidden_size).to(device)
        self.target_q_net2 = QNetworkFC(state_dim, action_dim, hidden_size).to(device)

        # Set the initial weight to be the same
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Optimizers
        self.q_optim1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optim2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)
        self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    @property
    def alpha(self):
        return torch.exp(self._alpha_param)

    def get_action(self, state, sample=True):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.float().to(self.device)

        dist = self.policy_net(state)

        if sample:
            action = dist.sample()
        else:
            action = dist.mean.detach()

        action = self.action_transform(action)

        return action.cpu().numpy()

    def store_transitions(self, state, action, reward, next_state, done):
        if len(state.shape) == 1 or state.shape[0] == 1:
            self.replay_buffer.add_transition(state, action, reward, next_state, done)
        else:
            self.replay_buffer.add_transitions(state, action, reward, next_state, done)

    def do_update(self):
        if len(self.replay_buffer) < self.min_transitions:
            return

        ## Preprocessing
        # Get some data and reshape it a little
        idx = list(np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False))
        mini_batch = self.replay_buffer[idx]

        states = mini_batch[0]
        actions = mini_batch[1]
        rewards = mini_batch[2]
        next_states = mini_batch[3]
        dones = mini_batch[4]

        #assert actions.shape == (self.batch_size, self.action_dim)
        #assert rewards.shape == (self.batch_size, )
        #assert states.shape == (self.batch_size, self.state_dim)
        #assert next_states.shape == (self.batch_size, self.state_dim)
        #assert dones.shape == (self.batch_size, )

        ## Q-network update
        # Compute Q-values with each net
        qval1 = self.q_net1(states, actions).squeeze(-1)
        qval2 = self.q_net2(states, actions).squeeze(-1)

        # Compute next state action distributions
        with torch.no_grad():
            next_state_act_dist = self.policy_net(next_states)

            # Sample next state actions
            next_actions = next_state_act_dist.sample()
            next_actions_tf = self.action_transform(next_actions)

            # Compute target Q-values with each net
            next_qval1 = self.target_q_net1(next_states, next_actions_tf).squeeze(-1)
            next_qval2 = self.target_q_net2(next_states, next_actions_tf).squeeze(-1)

            # Compute targets
            next_act_logprobs = next_state_act_dist.log_prob(next_actions)

            # Correct logprobs with the transformation
            next_act_logprobs -= self.action_transform.log_abs_det_jacobian(next_actions, next_actions_tf)
            next_act_logprobs = next_act_logprobs.sum(dim=-1)

            # Paper formula for comparison
            #next_act_logprobs_ppr = next_state_act_dist.log_prob(next_actions).sum(dim=-1)
            #next_act_logprobs_ppr -= torch.log(1-torch.tanh(next_actions)**2).sum(dim=-1)

            next_q = (1-dones) * (torch.min(next_qval1, next_qval2) - self.alpha * next_act_logprobs)

            q_target = rewards + self.gamma * next_q

        #assert next_q.shape == (self.batch_size, )
        #assert next_act_logprobs.shape == (self.batch_size, )
        #assert q_target.shape == qval1.shape == (self.batch_size, )
        #assert q_target.shape == qval2.shape

        q_loss_1 = F.mse_loss(qval1, q_target)
        q_loss_2 = F.mse_loss(qval2, q_target)

        # Update Qs
        self.q_optim1.zero_grad()
        self.q_optim2.zero_grad()

        q_loss_1.backward()
        q_loss_2.backward()

        self.q_optim1.step()
        self.q_optim2.step()

        ## Policy update
        state_act_dist = self.policy_net(states)
        sampled_actions = state_act_dist.rsample()

        sampled_actions_tf = self.action_transform(sampled_actions)
        #sampled_a_logprobs = state_act_dist.log_prob(sampled_actions)

        # Correct logprobs with the transformation
        #sampled_a_logprobs -= self.action_transform.log_abs_det_jacobian(sampled_actions, sampled_actions_tf)
        #sampled_a_logprobs = sampled_a_logprobs.sum(dim=-1)

        #sampled_a_logprobs_ppr = state_act_dist.log_prob(sampled_actions).sum(dim=-1)
        #sampled_a_logprobs_ppr -= torch.log(1-torch.tanh(sampled_actions)**2).sum(dim=-1)

        sampled_a_logprobs = state_act_dist.log_prob(sampled_actions).sum(dim=-1)
        sampled_a_logprobs -= (2*(np.log(2) - sampled_actions- F.softplus(-2*sampled_actions))).sum(axis=1)

        # Compute Q-values for the sampled actions
        qval1_a = self.q_net1(states, sampled_actions_tf).squeeze(-1)
        qval2_a = self.q_net2(states, sampled_actions_tf).squeeze(-1)

        #assert qval1_a.shape == (self.batch_size, )
        #assert qval2_a.shape == (self.batch_size, )

        # Update policy
        policy_obj_q = torch.min(qval1_a, qval2_a)
        policy_obj_entr = -self.alpha * sampled_a_logprobs

        policy_obj = policy_obj_q + policy_obj_entr

        #assert policy_obj.shape == (self.batch_size, )

        # Flip signs and reduce
        policy_obj = -policy_obj.mean()
        self.policy_optim.zero_grad()
        policy_obj.backward()
        self.policy_optim.step()

        # Update target networks
        self.target_q_net1.do_polyak(self.q_net1, self.polyak)
        self.target_q_net2.do_polyak(self.q_net2, self.polyak)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("q_loss/q1_loss", q_loss_1, self.num_updates)
            self.tb_writer.add_scalar("q_loss/q2_loss", q_loss_2, self.num_updates)
            self.tb_writer.add_scalar("policy_loss/total", policy_obj, self.num_updates)
            self.tb_writer.add_scalar("policy_loss/policy_obj_q", policy_obj_q.mean(), self.num_updates)
            self.tb_writer.add_scalar("policy_loss/policy_obj_H", policy_obj_entr.mean(), self.num_updates)
            #self.tb_writer.add_histogram("actions", sampled_actions_tf, self.num_updates)
            self.tb_writer.add_scalar("alpha", self.alpha, self.num_updates)
            self.tb_writer.add_scalar("log_probs", sampled_a_logprobs.mean(), self.num_updates)
            self.tb_writer.add_scalar("mean_std", state_act_dist.stddev.mean(), self.num_updates)

        if self.alpha_optim:
            alpha_loss = -self.alpha * (sampled_a_logprobs.mean().detach() + self.target_entropy)
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        self.num_updates += 1


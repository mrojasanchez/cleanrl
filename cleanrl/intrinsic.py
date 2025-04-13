import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# @title Replay buffer
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):

        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            # Use torch tensors directly if possible, otherwise convert carefully
            obses_t.append(np.asarray(obs_t))
            actions.append(np.asarray(action))
            # Check if reward is already a scalar or needs .item()
            rewards.append(reward if isinstance(reward, (int, float, np.number)) else reward.item())
            obses_tp1.append(np.asarray(obs_tp1))
            dones.append(done) # Assuming done is boolean or 0/1

        # Convert lists of numpy arrays to single numpy arrays
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones).astype(np.float32) # Ensure dones are float for potential masking
        )

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)]
        return self._encode_sample(idxes)

class BaseIntrinsicRewardModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_intrinsic_reward(self, state, action, next_state):
        # Return shape should be (batch_size,) or match num_envs
        raise NotImplementedError

    def get_loss(self, state_batch, action_batch, next_state_batch):
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))

        def init_weights(tensor):
            if isinstance(tensor, nn.Linear):
                nn.init.xavier_uniform_(tensor.weight)
                if tensor.bias is not None:
                    nn.init.zeros_(tensor.bias) # Initialize bias too
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class Embedder(nn.Module):
    def __init__(self, states_size, embedding_size, hidden_size):
        super().__init__()
        self.module = MLP(states_size,
                          embedding_size, # Should this be hidden_size? Naming seems swapped
                          hidden_size)    # Should this be embedding_size?
    def forward(self, s):
        # Ensure input is float
        return self.module(s.float())


class GaussianForwardDynamics(nn.Module):
    def __init__(self, encoding_dim, action_size, latent_dim): # Changed action_size param
        super().__init__()
        # Use action_size parameter here
        self.fc = nn.Linear(encoding_dim + action_size, encoding_dim)
        self.fc_mu = nn.Linear(encoding_dim, latent_dim)
        self.fc_log_var = nn.Linear(encoding_dim, latent_dim)

        # Initialize weights
        def init_weights(m):
             if isinstance(m, nn.Linear):
                 nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, latent_state, action):
        # Ensure action is float and has correct dimensions if needed
        # Assuming action is already (batch_size, action_dim)
        # Ensure latent_state is also float
        x  = torch.cat([latent_state.float(), action.float()], dim=-1)
        x  = F.gelu(self.fc(x)) # Using gelu as in original, could be ReLU
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, -10, 2)
        return mu, log_var


class SurprisalModule(BaseIntrinsicRewardModule):
    def __init__(self, input_size, action_size, linear_size, hidden_size, *args, **kwargs):
        super().__init__()

        # Assuming linear_size is the first layer's hidden size and hidden_size is the embedding output size
        self.embedder = Embedder(input_size,
                                 linear_size, # Hidden size for MLP
                                 hidden_size) # Output embedding size

        self.forward_model = GaussianForwardDynamics(hidden_size, # Input embedding size
                                                     action_size, # Actual action size
                                                     hidden_size) # Output prediction size (matches embedding)

        # Remove eta0 and n, normalization can be handled externally or differently
        # self.n = 1
        # self.eta0 = 0

    def get_loss_vec(self, state_batch, action_batch, next_state_batch):
        # Ensure inputs are float tensors
        state_batch = state_batch.float()
        action_batch = action_batch.float()
        next_state_batch = next_state_batch.float()

        phi_s_batch = self.embedder(state_batch)

        with torch.no_grad():
            phi_next_s_batch = self.embedder(next_state_batch)
            # Detach to prevent gradients flowing back from target
            phi_next_s_batch = phi_next_s_batch.detach()

        mu, log_var = self.forward_model.forward(phi_s_batch, action_batch)
        
        # Ensure numerical stability for distribution
        var = torch.exp(log_var) + 1e-6 # Add small epsilon
        # dist = torch.distributions.MultivariateNormal(mu, torch.diag_embed(var)) # More stable than exp(log_var) directly
        # Use Normal and sum log_prob for independence assumption, common in practice
        dist = torch.distributions.Normal(mu, var.sqrt())
        # Calculate log prob per dimension and sum
        loss = -dist.log_prob(phi_next_s_batch).sum(dim=-1) # Sum over embedding dimension

        # loss = -dist.log_prob(phi_next_s_batch) # Log prob for MultivariateNormal

        return loss # Return vector of losses, shape (batch_size,)

    def get_intrinsic_reward(self, state, action, next_state):
        # Directly return the prediction error (negative log likelihood)
        # This is the "surprise" signal. Scaling happens outside this function.
        with torch.no_grad():
            loss_vec = self.get_loss_vec(state, action, next_state)
            # Remove the normalization part for now, handle scaling via weight
            # intrinsic_reward = self.normalise_reward(loss_vec)
        # Return the full vector, shape (num_envs,)
        return loss_vec # No [0] indexing

    def get_loss(self, state_batch, action_batch, next_state_batch, *args, **kwargs):
        loss_vec = self.get_loss_vec(state_batch, action_batch, next_state_batch)
        loss = torch.mean(loss_vec) # Average loss over the batch
        return loss

    # def normalise_reward(self, rewards_batch):
    #     if rewards_batch.shape[0] == 1:
    #         return rewards_batch

    #     mean_rewards = torch.abs(torch.mean(rewards_batch).view(rewards_batch.shape[0], 1))
    #     norm_rewards = (rewards_batch - torch.min([0, mean_rewards])) / torch.max([1, mean_rewards.squeeze()] )
    #     return norm_rewards
    
    # Keep normalization method if you want to use it later, but comment out its use for now
    # def normalise_reward(self, rewards_batch):
    #     # This simple normalization might be unstable. Consider running mean/std.
    #     if rewards_batch.shape[0] <= 1: # Handle single element batch
    #         return rewards_batch
    #
    #     mean_rewards = torch.mean(rewards_batch)
    #     std_rewards = torch.std(rewards_batch) + 1e-8 # Add epsilon for stability
    #
    #     # Normalize using mean and std
    #     norm_rewards = (rewards_batch - mean_rewards) / std_rewards
    #
    #     # Clip or scale as needed, simple normalization here
    #     return norm_rewards
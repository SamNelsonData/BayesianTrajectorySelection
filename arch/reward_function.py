"""
Reward models for IRD-RLHF.

Provides:
- BayesianRewardModel: IRD posterior over linear reward weights
- NeuralRewardModel: Trainable neural network reward (for preference learning)
- NeuralRewardEnsemble: Ensemble for uncertainty (Christiano et al. approach)
- Utility functions for computing true/proxy rewards
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy



class BayesianRewardModel:
    """
    Bayesian reward model for Inverse Reward Design.
    
    Maintains a posterior distribution over reward function weights:
    P(w_true | preferences) ∝ P(preferences | w_true) × P(w_true)
    
    Uses linear reward: r(trajectory) = w^T φ(trajectory)
    """
    
    FEATURE_NAMES = [
        "seeds_collected",
        "time_under_clouds",
        "battery_depletions",
        "avg_movement",
        "final_battery"
    ]
    
    def __init__(self, feature_dim=5, num_samples=100, prior_std=2.0):
        """
        @param feature_dim: dimension of feature vector
        @param num_samples: number of posterior samples to maintain
        @param prior_std: standard deviation of Gaussian prior
        """
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.prior_std = prior_std
        
        # Prior will be set when proxy_weights are provided
        self.prior_mean = None
        self.prior_cov = np.eye(feature_dim) * (prior_std ** 2)
        
        # Posterior samples (initialized after proxy weights are set)
        self.posterior_samples = None
        
        # Proxy reward weights (set by user)
        self.proxy_weights = None
    
    def set_proxy_reward(self, weights):
        """
        Set the proxy reward weights and initialize prior around them.
        
        FIX: Prior is now centered at proxy_weights, not zero!
        This means we start with beliefs close to the proxy reward.
        
        @param weights: array of shape (feature_dim,)
        """
        self.proxy_weights = np.array(weights, dtype=np.float32)
        assert len(self.proxy_weights) == self.feature_dim
        
        # FIX: Center prior around proxy weights
        self.prior_mean = self.proxy_weights.copy()
        
        # Initialize posterior samples from prior (centered at proxy)
        self.posterior_samples = np.random.multivariate_normal(
            self.prior_mean,
            self.prior_cov,
            size=self.num_samples
        )
    
    def compute_features(self, world, trajectory):
        """
        Extract feature vector from a trajectory.
        
        IMPORTANT: Does NOT modify world state - uses deep copy.
        
        Features:
        1. Seeds collected (positive is good)
        2. Time spent under clouds (negative is good -> want negative weight)
        3. Battery depletion events (negative is good -> want negative weight)
        4. Average movement per step (can be positive or negative)
        5. Final battery level normalized (positive is good)
        
        @param world: SciWrld instance
        @param trajectory: list of (row, col) positions
        @return: feature vector of shape (feature_dim,)
        """
        # Work on a copy to avoid side effects
        world_copy = deepcopy(world)
        
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        if len(trajectory) == 0:
            return features
        
        seeds_collected = 0
        time_under_clouds = 0
        battery_depletions = 0
        total_distance = 0
        battery = 2  # Starting battery
        
        prev_pos = trajectory[0]
        
        for pos in trajectory:
            pos = tuple(pos)
            
            # Feature 1: Seed collection
            if world_copy.get_cell(pos) == world_copy.SEED:
                seeds_collected += 1
                world_copy.world[pos] = world_copy.EMPTY  # Consume seed in copy
            
            # Feature 2: Cloud exposure
            if world_copy.is_under_cloud(pos):
                time_under_clouds += 1
                battery -= 1
            elif battery < 2:
                battery += 1  # Recharge in sunlight
            
            # Feature 3: Battery depletion
            if battery <= 0:
                battery_depletions += 1
                battery = 0  # Can't go negative
            
            # Feature 4: Distance traveled
            total_distance += abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
            prev_pos = pos
        
        # Compute final features
        features[0] = seeds_collected
        features[1] = time_under_clouds
        features[2] = battery_depletions
        features[3] = total_distance / len(trajectory) if len(trajectory) > 1 else 0
        features[4] = max(0, battery) / 2.0  # Normalized final battery
        
        return features
    
    def compute_reward(self, features, weights):
        """
        Compute reward for features given weights.
        
        @param features: feature vector
        @param weights: weight vector
        @return: scalar reward
        """
        return np.dot(weights, features)
    
    def compute_reward_uncertainty(self, world, trajectory):
        """
        Compute mean reward and variance across posterior samples.
        
        This is the key for trajectory selection: high variance = high uncertainty.
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @return: (mean_reward, variance)
        """
        features = self.compute_features(world, trajectory)
        
        # Compute reward under each posterior sample
        rewards = np.array([
            self.compute_reward(features, w)
            for w in self.posterior_samples
        ])
        
        return np.mean(rewards), np.var(rewards)
    
    def update_posterior_pairwise(self, traj1, world1, traj2, world2, preferred_idx, temperature=1.0):
        """
        FIX: Update posterior using PAIRWISE preference (Bradley-Terry model).
        
        This is the correct way to learn from preferences!
        
        Likelihood: P(traj_i > traj_j | w) = σ((r_w(traj_i) - r_w(traj_j)) / τ)
        where σ is the sigmoid function.
        
        @param traj1: first trajectory
        @param world1: world state for traj1
        @param traj2: second trajectory  
        @param world2: world state for traj2
        @param preferred_idx: 0 if traj1 preferred, 1 if traj2 preferred
        @param temperature: softmax temperature
        """
        if self.proxy_weights is None:
            raise ValueError("Must set proxy_weights before updating posterior")
        
        # Compute features for both trajectories
        features1 = self.compute_features(world1, traj1)
        features2 = self.compute_features(world2, traj2)
        
        # Feature difference: positive if preferred is traj1, negative if traj2
        if preferred_idx == 0:
            # traj1 preferred
            feat_diff = features1 - features2
        else:
            # traj2 preferred
            feat_diff = features2 - features1
        
        # Compute log-likelihood for each posterior sample using Bradley-Terry
        log_likelihoods = []
        
        for w_candidate in self.posterior_samples:
            # Reward difference (positive means preferred traj has higher reward)
            reward_diff = np.dot(w_candidate, feat_diff)
            
            # Bradley-Terry log-likelihood: log(sigmoid(reward_diff / temperature))
            # = -log(1 + exp(-reward_diff / temperature))
            scaled_diff = reward_diff / temperature
            
            # Numerically stable log-sigmoid
            if scaled_diff > 0:
                log_likelihood = -np.log1p(np.exp(-scaled_diff))
            else:
                log_likelihood = scaled_diff - np.log1p(np.exp(scaled_diff))
            
            log_likelihoods.append(log_likelihood)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Importance resampling
        weights = np.exp(log_likelihoods - np.max(log_likelihoods))
        weights /= weights.sum()
        
        # Check for weight collapse
        effective_samples = 1.0 / np.sum(weights ** 2)
        
        # Resample
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=weights,
            replace=True
        )
        self.posterior_samples = self.posterior_samples[indices].copy()
        
        # Add small noise to maintain diversity (prevent collapse)
        # Scale noise based on effective sample size
        noise_scale = 0.05 * (1.0 - effective_samples / self.num_samples)
        noise = np.random.multivariate_normal(
            np.zeros(self.feature_dim),
            np.eye(self.feature_dim) * noise_scale,
            size=self.num_samples
        )
        self.posterior_samples += noise
    
    def update_posterior(self, preferred_trajectory, world, temperature=1.0):
        """
        DEPRECATED: Single-trajectory update (keeps for compatibility but warns).
        
        This method is kept for backward compatibility but should not be used.
        Use update_posterior_pairwise instead!
        """
        import warnings
        warnings.warn(
            "update_posterior(single trajectory) is deprecated. "
            "Use update_posterior_pairwise(traj1, world1, traj2, world2, preferred_idx) instead.",
            DeprecationWarning
        )
        
        # Fall back to old behavior (biased toward positive weights)
        if self.proxy_weights is None:
            raise ValueError("Must set proxy_weights before updating posterior")
        
        features = self.compute_features(world, preferred_trajectory)
        
        log_likelihoods = []
        for w_candidate in self.posterior_samples:
            candidate_reward = self.compute_reward(features, w_candidate)
            log_likelihood = candidate_reward / temperature
            log_likelihoods.append(log_likelihood)
        
        log_likelihoods = np.array(log_likelihoods)
        weights = np.exp(log_likelihoods - np.max(log_likelihoods))
        weights /= weights.sum()
        
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=weights,
            replace=True
        )
        self.posterior_samples = self.posterior_samples[indices].copy()
        
        noise = np.random.multivariate_normal(
            np.zeros(self.feature_dim),
            np.eye(self.feature_dim) * 0.01,
            size=self.num_samples
        )
        self.posterior_samples += noise
    
    def get_mean_weights(self):
        """Get posterior mean of weights."""
        return np.mean(self.posterior_samples, axis=0)
    
    def get_std_weights(self):
        """Get posterior standard deviation of weights."""
        return np.std(self.posterior_samples, axis=0)

class NeuralRewardModel(nn.Module):
    """
    Trainable neural network reward model.
    
    Used with PreferenceLearner to learn from human preferences.
    """
    
    def __init__(self, state_dim=7, hidden_dim=64):
        """
        @param state_dim: dimension of encoded state
        @param hidden_dim: hidden layer size
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states):
        """
        Compute reward for each state.
        
        @param states: tensor of shape (batch, state_dim) or (state_dim,)
        @return: rewards of shape (batch, 1) or (1,)
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        return self.net(states)
    
    def predict_trajectory_reward(self, trajectory_states):
        """
        Compute total reward for a trajectory.
        
        @param trajectory_states: array/tensor of shape (T, state_dim)
        @return: scalar tensor (for backprop)
        """
        if not isinstance(trajectory_states, torch.Tensor):
            trajectory_states = torch.tensor(trajectory_states, dtype=torch.float32)
        
        if trajectory_states.dim() == 1:
            trajectory_states = trajectory_states.unsqueeze(0)
        
        return self.forward(trajectory_states).sum()


class NeuralRewardEnsemble:
    """
    Ensemble of neural reward models for uncertainty estimation.
    
    This is the approach from Christiano et al. (2017):
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, state_dim=7, hidden_dim=64, num_models=5):
        """
        @param state_dim: input dimension
        @param hidden_dim: hidden layer size
        @param num_models: number of ensemble members
        """
        self.num_models = num_models
        self.models = nn.ModuleList([
            NeuralRewardModel(state_dim, hidden_dim)
            for _ in range(num_models)
        ])
    
    def predict_trajectory_reward(self, trajectory_states):
        """
        Predict reward with uncertainty.
        
        @param trajectory_states: array of shape (T, state_dim)
        @return: (mean_reward, variance)
        """
        if not isinstance(trajectory_states, torch.Tensor):
            trajectory_states = torch.tensor(trajectory_states, dtype=torch.float32)
        
        rewards = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                r = model.predict_trajectory_reward(trajectory_states).item()
                rewards.append(r)
        
        return np.mean(rewards), np.var(rewards)
    
    def parameters(self):
        """Return all parameters for optimizer."""
        return self.models.parameters()
    
    def train(self, mode=True):
        """Set training mode."""
        for model in self.models:
            model.train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        self.train(False)

def compute_true_reward(world, trajectory):
    """
    Compute the TRUE reward for a trajectory.
    
    This is what we want the agent to actually optimize.
    Includes safety constraints that proxy might miss.
    
    IMPORTANT: Does NOT modify world state.
    
    @param world: SciWrld instance
    @param trajectory: list of positions
    @return: total reward (float)
    """
    world_copy = deepcopy(world)
    
    total_reward = 0.0
    battery = 2
    
    for pos in trajectory:
        pos = tuple(pos)
        step_reward = -0.1  # Living penalty
        
        # Seed collection
        if world_copy.get_cell(pos) == world_copy.SEED:
            step_reward += 10.0
            world_copy.world[pos] = world_copy.EMPTY
        
        # Cloud penalty
        if world_copy.is_under_cloud(pos):
            step_reward -= 5.0
            battery -= 1
        elif battery < 2:
            battery += 1
        
        # Battery depletion penalty
        if battery <= 0:
            step_reward -= 20.0
        
        total_reward += step_reward
    
    return total_reward


def compute_proxy_reward(world, trajectory):
    """
    Compute PROXY reward (simplified version designer might use).
    
    This is intentionally simpler than true reward - it might miss
    important safety constraints like battery management.
    
    @param world: SciWrld instance
    @param trajectory: list of positions
    @return: total reward (float)
    """
    world_copy = deepcopy(world)
    
    total_reward = 0.0
    
    for pos in trajectory:
        pos = tuple(pos)
        step_reward = -0.1  # Living penalty
        
        # Proxy only cares about seeds (ignores clouds/battery!)
        if world_copy.get_cell(pos) == world_copy.SEED:
            step_reward += 10.0
            world_copy.world[pos] = world_copy.EMPTY
        
        total_reward += step_reward
    
    return total_reward


def encode_trajectory_states(world, trajectory):
    """
    Convert trajectory to encoded states for neural models.
    
    IMPORTANT: Does NOT modify world state.
    
    @param world: SciWrld instance
    @param trajectory: list of positions
    @return: array of shape (len(trajectory), state_dim)
    """
    from arch.sciwrld import encode_state
    
    # Work on copy to avoid side effects
    world_copy = deepcopy(world)
    
    states = []
    for pos in trajectory:
        state = encode_state(world_copy, pos)
        states.append(state)
    
    return np.array(states, dtype=np.float32)
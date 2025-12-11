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
    P(w_true | w_proxy, M) ∝ P(w_proxy | w_true, M) × P(w_true)
    
    Uses linear reward: r(trajectory) = w^T φ(trajectory)
    where φ extracts trajectory features.
    """
    
    FEATURE_NAMES = [
        "seeds_collected",
        "time_under_clouds",
        "battery_depletions",
        "avg_movement",
        "final_battery"
    ]
    
    def __init__(self, feature_dim=5, num_samples=100, prior_std=1.0):
        """
        @param feature_dim: dimension of feature vector
        @param num_samples: number of posterior samples to maintain
        @param prior_std: standard deviation of Gaussian prior
        """
        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.prior_std = prior_std
        
        # Prior: N(0, prior_std^2 * I)
        self.prior_mean = np.zeros(feature_dim)
        self.prior_cov = np.eye(feature_dim) * (prior_std ** 2)
        
        # Initialize posterior samples from prior
        self.posterior_samples = np.random.multivariate_normal(
            self.prior_mean,
            self.prior_cov,
            size=num_samples
        )
        
        # Proxy reward weights (set by user)
        self.proxy_weights = None
    
    def set_proxy_reward(self, weights):
        """
        Set the proxy reward weights (what designer gave to agent).
        
        @param weights: array of shape (feature_dim,)
        """
        self.proxy_weights = np.array(weights, dtype=np.float32)
        assert len(self.proxy_weights) == self.feature_dim
    
    def compute_features(self, world, trajectory):
        """
        Extract feature vector from a trajectory.
        
        IMPORTANT: Does NOT modify world state - uses deep copy.
        
        Features:
        1. Seeds collected
        2. Time spent under clouds
        3. Battery depletion events
        4. Average movement per step
        5. Final battery level (normalized)
        
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
    
    def update_posterior(self, preferred_trajectory, world, temperature=1.0):
        """
        Update posterior based on observed human preference.
        
        Uses importance sampling: trajectories preferred by humans should have
        higher reward under the true reward function.
        
        @param preferred_trajectory: trajectory that human preferred
        @param world: SciWrld instance
        @param temperature: softmax temperature for likelihood
        """
        if self.proxy_weights is None:
            raise ValueError("Must set proxy_weights before updating posterior")
        
        # Compute features for preferred trajectory
        features = self.compute_features(world, preferred_trajectory)
        
        # Compute log-likelihood for each posterior sample
        # Higher reward under candidate weights → more likely to be true weights
        log_likelihoods = []
        
        for w_candidate in self.posterior_samples:
            candidate_reward = self.compute_reward(features, w_candidate)
            # Likelihood: softmax over reward
            log_likelihood = candidate_reward / temperature
            log_likelihoods.append(log_likelihood)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Importance resampling
        weights = np.exp(log_likelihoods - np.max(log_likelihoods))
        weights /= weights.sum()
        
        # Resample
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=weights,
            replace=True
        )
        self.posterior_samples = self.posterior_samples[indices].copy()
        
        # Add small noise to maintain diversity (prevent collapse)
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


# =============================================================================
# Reward Functions (no side effects!)
# =============================================================================

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
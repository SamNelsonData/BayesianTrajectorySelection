import torch
import torch.nn as nn
import numpy as np
from arch.sciwrld import SciWrld
from scipy.special import softmax
from copy import deepcopy

class BayesianRewardModel:
    """
    Bayesian reward model for Inverse Reward Design.
    
    Maintains a posterior distribution over reward function weights:
    P(w_true | w_proxy, M) ∝ P(w_proxy | w_true, M) × P(w_true)
    
    For simplicity, we use a linear reward model:
    r(ξ; w) = w^T φ(ξ)
    
    where φ(ξ) are trajectory features.
    """
    
    def __init__(self, feature_dim=5, num_samples=100, prior_std=1.0):

        self.feature_dim = feature_dim
        self.num_samples = num_samples
        self.prior_std = prior_std
        
        # Prior: Gaussian distribution on weights
        # P(w_true) = N(0, prior_std^2 * I)
        self.prior_mean = np.zeros(feature_dim)
        self.prior_cov = np.eye(feature_dim) * (prior_std ** 2)
        
        # Posterior samples: list of weight vectors
        # Initially sample from prior
        self.posterior_samples = np.random.multivariate_normal(
            self.prior_mean,
            self.prior_cov,
            size=num_samples
        )
        
        # Track proxy reward for inference
        self.proxy_weights = None
    
    def set_proxy_reward(self, proxy_weights):
        """
        Set the observed proxy reward weights.
        This is what the designer gave to the agent.
        
        @param proxy_weights: weight vector for proxy reward [feature_dim]
        """
        self.proxy_weights = np.array(proxy_weights)
    
    def compute_features(self, world: SciWrld, trajectory):
        """
        Extract feature vector φ(ξ) from a trajectory.
        
        Features:
        1. Number of seeds collected
        2. Time spent under clouds
        3. Battery depletion events
        4. Total distance traveled
        5. Final battery level
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @return: feature vector [feature_dim]
        """
        from copy import deepcopy
        
        # Make a copy to avoid modifying the original world
        world_copy = deepcopy(world)
        
        features = np.zeros(self.feature_dim)
        
        seeds_collected = 0
        time_under_clouds = 0
        battery_depletions = 0
        distance = 0
        battery = 2  # Starting battery
        
        prev_pos = trajectory[0] if trajectory else (0, 0)
        
        for pos in trajectory:
            # Feature 1: Seeds collected (check BEFORE consuming)
            try:
                if world_copy.world[pos].item() == world_copy.item_to_value['Seed']:
                    seeds_collected += 1
                    world_copy.world[pos] = world_copy.item_to_value['Sand']  # Consume seed
            except:
                import pdb; pdb.set_trace()

            
            # Feature 2: Time under clouds
            under_cloud = False
            for cloud, _ in world_copy.clouds:
                if pos in cloud:
                    under_cloud = True
                    time_under_clouds += 1
                    battery -= 1
                    break
            
            if not under_cloud and battery < 2:
                battery = battery + 1
            
            # Feature 3: Battery depletions
            if battery <= 0:
                battery_depletions += 1
            
            # Feature 4: Distance traveled
            distance += abs(pos[0] - prev_pos[0]) + abs(pos[1] - prev_pos[1])
            prev_pos = pos
        
        # Feature 5: Final battery
        final_battery = battery
        
        features[0] = seeds_collected
        features[1] = time_under_clouds
        features[2] = battery_depletions
        features[3] = distance / len(trajectory) if trajectory else 0.0  # Normalized
        features[4] = final_battery / 2.0  # Normalized
        
        return features
    
    def reward(self, features, weights):
        """
        Compute reward for trajectory features given weights.
        r(ξ; w) = w^T φ(ξ)
        
        @param features: feature vector [feature_dim]
        @param weights: weight vector [feature_dim]
        @return: scalar reward
        """
        return np.dot(weights, features)
    
    def update_posterior(self, observed_trajectories, world, temperature=1.0):
        """
        Update posterior distribution using observed agent behavior.
        
        Uses the likelihood:
        P(w_proxy | w_true, M) ∝ exp(R(π*_proxy) / R(π*_true))
        
        where π*_proxy is the optimal policy under proxy reward,
        and π*_true is optimal under true reward.
        
        @param observed_trajectories: list of trajectories from agent
        @param world: SciWrld instance
        @param temperature: temperature for likelihood computation
        """
        if self.proxy_weights is None:
            raise ValueError("Must set proxy_weights before updating posterior")
        
        # Compute features for observed trajectories
        observed_features = []
        for traj in observed_trajectories:
            features = self.compute_features(world, traj)
            observed_features.append(features)
        
        observed_features = np.array(observed_features)
        
        # Compute likelihood weights for each posterior sample
        # P(w_proxy | w_candidate) measures how likely w_proxy was chosen
        # if the true reward had weights w_candidate
        
        log_likelihoods = []
        
        for w_candidate in self.posterior_samples:
            # Compute proxy reward and candidate reward for observed trajectories
            proxy_rewards = np.array([
                self.reward(feat, self.proxy_weights) 
                for feat in observed_features
            ])
            
            candidate_rewards = np.array([
                self.reward(feat, w_candidate)
                for feat in observed_features
            ])
            
            # Likelihood: how well does w_candidate explain choosing w_proxy?
            # If w_candidate predicts high reward for trajectories that w_proxy
            # also rates highly, then w_candidate is more likely
            
            # Correlation-based likelihood
            if len(proxy_rewards) > 1:
                correlation = np.corrcoef(proxy_rewards, candidate_rewards)[0, 1]
                log_likelihood = correlation / temperature
            else:
                # For single trajectory, use reward similarity
                reward_diff = np.abs(proxy_rewards[0] - candidate_rewards[0])
                log_likelihood = -reward_diff / temperature
            
            log_likelihoods.append(log_likelihood)
        
        log_likelihoods = np.array(log_likelihoods)
        
        # Importance resampling
        # Compute normalized weights
        weights = np.exp(log_likelihoods - np.max(log_likelihoods))
        weights /= weights.sum()
        
        # Resample according to importance weights
        indices = np.random.choice(
            self.num_samples,
            size=self.num_samples,
            p=weights,
            replace=True
        )
        
        self.posterior_samples = self.posterior_samples[indices]
        
        # Add small noise to maintain diversity
        noise = np.random.multivariate_normal(
            np.zeros(self.feature_dim),
            np.eye(self.feature_dim) * 0.01,
            size=self.num_samples
        )
        self.posterior_samples += noise
    
    def compute_reward_uncertainty(self, world: SciWrld, trajectory):
        """
        Compute variance in predicted reward across posterior samples.
        This is the key for trajectory selection in RLHF.
        
        V(ξ) = Var[r(ξ; w) | w ~ P(w_true | w_proxy, M)]
        
        @param world: SciWrld instance  
        @param trajectory: trajectory to evaluate
        @return: (mean_reward, reward_variance)
        """
        features = self.compute_features(world, trajectory)
        
        # Compute reward under each posterior sample
        rewards = np.array([
            self.reward(features, w_sample)
            for w_sample in self.posterior_samples
        ])
        
        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        
        return mean_reward, reward_variance
    
    def get_mean_weights(self):
        """
        Get the posterior mean of the reward weights.
        
        @return: mean weight vector [feature_dim]
        """
        return np.mean(self.posterior_samples, axis=0)
    
    def get_reward_std(self):
        """
        Get the posterior standard deviation of each weight.
        
        @return: std vector [feature_dim]
        """
        return np.std(self.posterior_samples, axis=0)


class NeuralRewardEnsemble:
    """
    Ensemble of neural network reward models for comparison with IRD.
    This is the approach from Christiano et al. (2017).
    
    Uncertainty is estimated as variance across ensemble members.
    """
    
    def __init__(self, state_dim=7, hidden_dim=64, num_models=5):
        """
        @param state_dim: dimension of encoded state
        @param hidden_dim: hidden layer size
        @param num_models: number of ensemble members
        """
        self.num_models = num_models
        self.models = []
        
        for _ in range(num_models):
            model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.models.append(model)
    
    def predict_trajectory_reward(self, trajectory_states):
        """
        Predict reward for a trajectory using all ensemble members.
        
        @param trajectory_states: encoded states [T, state_dim]
        @return: (mean_reward, reward_variance)
        """
        if isinstance(trajectory_states, list):
            trajectory_states = np.array(trajectory_states)
        
        states_tensor = torch.tensor(trajectory_states, dtype=torch.float32)
        
        rewards = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                r = model(states_tensor).sum().item()
                rewards.append(r)
        
        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        
        return mean_reward, reward_variance


def proxy_reward_function(world: SciWrld, position):
    """
    Proxy reward function (what designer gives to agent).
    Simplified version that might miss important safety constraints.
    
    @param world: SciWrld instance
    @param position: agent position
    @return: reward value
    """
    reward = 0.0
    row, col = position
    
    # Proxy only cares about collecting seeds
    if world.world[row, col] == world.item_to_value['Seed']:
        reward += 10.0
    
    # Small movement penalty
    reward -= 0.1
    
    return reward


def true_reward_function(world: SciWrld, position):
    """
    True reward function (what designer actually wanted).
    Includes safety constraints the proxy might miss.
    
    @param world: SciWrld instance
    @param position: agent position
    @return: reward value
    """
    reward = -0.1  # Living penalty
    row, col = position
    
    # Collect seeds
    if world.world[row, col] == world.item_to_value['Seed']:
        reward += 10.0
    
    # Avoid clouds (battery management)
    under_cloud = False
    for cloud, _ in world.clouds:
        if (row, col) in cloud:
            under_cloud = True
            break
    
    if under_cloud:
        reward -= 5.0
        world.agent.battery -= 1
    else:
        world.agent.battery = min(2, world.agent.battery + 1)
    
    # Battery depletion penalty
    if world.agent.battery <= 0:
        reward -= 20.0
    
    return reward


def encode_trajectory_states(world: SciWrld, trajectory):
    """
    Convert trajectory positions to encoded states.
    
    @param world: SciWrld instance
    @param trajectory: list of positions
    @return: array of shape [len(trajectory), state_dim]
    """
    from arch.sciwrld import encode_state
    
    orig_pos = world.agent.position
    orig_battery = world.agent.battery
    
    states = []
    for pos in trajectory:
        world.agent.position = pos
        state = encode_state(world)
        states.append(state)
        world.world_time += 1
    
    world.agent.position = orig_pos
    world.agent.battery = orig_battery
    
    return np.array(states)
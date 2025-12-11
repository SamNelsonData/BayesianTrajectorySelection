"""
Policy Network and Trainer for RLHF.

Provides:
- PolicyNetwork: Neural network that outputs action probabilities
- PolicyTrainer: Trains policy to maximize learned reward (REINFORCE, A2C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy


class PolicyNetwork(nn.Module):
    """
    Policy network for discrete actions.
    
    Outputs probability distribution over actions:
    0=Up, 1=Left, 2=Down, 3=Right
    """
    
    def __init__(self, input_dim=7, hidden_dim=64, num_actions=4):

        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self.num_actions = num_actions
    
    def forward(self, state):
        """
        Compute action probabilities.
        
        @param state: tensor of shape (batch, input_dim) or (input_dim,)
        @return: probabilities of shape (batch, num_actions)
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        logits = self.net(state)
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        @param state: state tensor
        @param deterministic: if True, return argmax action
        @return: (action, log_prob)
        """
        probs = self.forward(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_action(self, state, action):
        """
        Evaluate log probability and entropy for a given action.
        
        @param state: state tensor
        @param action: action tensor
        @return: (log_prob, entropy)
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy


class PolicyTrainer:
    """
    Trains policy to maximize learned reward.
    
    Implements REINFORCE (vanilla policy gradient).
    """
    
    def __init__(
        self,
        policy,
        reward_model,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01
    ):
        """
        @param policy: PolicyNetwork to train
        @param reward_model: learned reward model (BayesianRewardModel or Neural)
        @param learning_rate: optimizer learning rate
        @param gamma: discount factor
        @param entropy_coef: entropy bonus coefficient
        """
        self.policy = policy
        self.reward_model = reward_model
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
    
    def train(
        self,
        env_class,
        env_kwargs,
        agent_class,
        num_episodes=1000,
        max_steps=20,
        verbose=True
    ):
        """
        Train policy using REINFORCE.
        
        @param env_class: SciWrld class
        @param env_kwargs: arguments for environment constructor
        @param agent_class: PolicyAgent class
        @param num_episodes: number of training episodes
        @param max_steps: max steps per episode
        @param verbose: print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING POLICY")
            print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
            print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Create fresh environment
            env = env_class(**env_kwargs)
            
            # Add some clouds
            for _ in range(3):
                env.step(new_cloud_rate=0.5, cloud_limit=3)
            
            # Add agent with our policy
            env.add_agent(agent_class, policy_network=self.policy)
            
            # Run episode and collect data
            episode_reward, episode_length = self._run_episode(env, max_steps)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                print(f"  Avg Length (last 100): {avg_length:.1f}")
        
        if verbose:
            print("\n✓ Policy training complete!")
    
    def _run_episode(self, env, max_steps):
        """
        Run one episode and update policy.
        
        @return: (total_reward, episode_length)
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        entropies = []
        
        for step in range(max_steps):
            # Get state
            from arch.sciwrld import encode_state
            state = encode_state(env)
            states.append(state)
            
            # Get action from policy
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, log_prob = self.policy.get_action(state_tensor)
            
            # Get entropy for bonus
            _, entropy = self.policy.evaluate_action(state_tensor, action)
            
            actions.append(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)
            
            # Take action
            env.agent.act(action.item())
            
            # Compute reward using LEARNED reward model
            reward = self._compute_learned_reward(env, env.agent.position)
            rewards.append(reward)
            
            # Update battery based on cloud exposure
            if env.is_under_cloud(env.agent.position):
                env.agent.battery -= 1
            elif env.agent.battery < 2:
                env.agent.battery += 1
            
            # Check termination
            if env.agent.battery <= 0:
                break
        
        # Update policy
        self._update_policy(log_probs, rewards, entropies)
        
        return sum(rewards), len(rewards)
    
    def _compute_learned_reward(self, env, position):
        """
        Compute reward using learned model (NOT true reward!).
        """
        trajectory = [position]
        
        # BayesianRewardModel
        if hasattr(self.reward_model, 'compute_reward_uncertainty'):
            mean_reward, _ = self.reward_model.compute_reward_uncertainty(env, trajectory)
            return mean_reward
        
        # NeuralRewardModel
        elif hasattr(self.reward_model, 'predict_trajectory_reward'):
            from arch.reward_model import encode_trajectory_states
            states = encode_trajectory_states(env, trajectory)
            states_tensor = torch.tensor(states, dtype=torch.float32)
            
            self.reward_model.eval()
            with torch.no_grad():
                reward = self.reward_model.predict_trajectory_reward(states_tensor).item()
            return reward
        
        else:
            # Fallback: simple reward
            reward = -0.1
            if env.get_cell(position) == env.SEED:
                reward += 10.0
            return reward
    
    def _update_policy(self, log_probs, rewards, entropies):
        """
        REINFORCE policy update.
        
        Loss = -∑ log π(a|s) * G_t + entropy_bonus
        """
        if len(rewards) == 0:
            return
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = []
        entropy_bonus = []
        
        for log_prob, G, ent in zip(log_probs, returns, entropies):
            policy_loss.append(-log_prob * G)
            entropy_bonus.append(-self.entropy_coef * ent)
        
        total_loss = torch.stack(policy_loss).sum() + torch.stack(entropy_bonus).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        self.loss_history.append(total_loss.item())
    
    def evaluate(self, env_class, env_kwargs, agent_class, num_episodes=10, max_steps=20):
        """
        Evaluate trained policy.
        
        @return: dict with metrics
        """
        print(f"\n{'='*60}")
        print("EVALUATING POLICY")
        print(f"{'='*60}\n")
        
        self.policy.eval()
        
        episode_rewards = []
        seeds_collected = []
        
        for ep in range(num_episodes):
            env = env_class(**env_kwargs)
            for _ in range(3):
                env.step(new_cloud_rate=0.5, cloud_limit=3)
            
            env.add_agent(agent_class, policy_network=self.policy)
            
            total_reward = 0
            seeds = 0
            
            for step in range(max_steps):
                old_cell = env.get_cell(env.agent.position)
                
                # Deterministic action
                env.agent.act(deterministic=True)
                
                # Check seed collection
                if old_cell == env.SEED:
                    seeds += 1
                
                # Compute reward
                reward = self._compute_learned_reward(env, env.agent.position)
                total_reward += reward
                
                # Update battery
                if env.is_under_cloud(env.agent.position):
                    env.agent.battery -= 1
                elif env.agent.battery < 2:
                    env.agent.battery += 1
                
                if env.agent.battery <= 0:
                    break
            
            episode_rewards.append(total_reward)
            seeds_collected.append(seeds)
            
            print(f"Episode {ep+1}: Reward={total_reward:.2f}, Seeds={seeds}")
        
        self.policy.train()
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_seeds': np.mean(seeds_collected),
            'max_seeds': np.max(seeds_collected)
        }
        
        print(f"\nResults:")
        print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean seeds: {results['mean_seeds']:.2f}")
        
        return results
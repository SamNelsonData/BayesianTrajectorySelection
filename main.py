import argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch

from arch.sciwrld import SciWrld
from arch.agents import Agent, PolicyAgent, trajectory_similarity
from arch.reward_function import (
    BayesianRewardModel,
    NeuralRewardModel,
    NeuralRewardEnsemble,
    compute_true_reward,
    encode_trajectory_states
)
from arch.preference_learning import (
    PreferenceDataset,
    PreferenceLearner,
    EnsemblePreferenceLearner
)
from arch.policy import PolicyNetwork, PolicyTrainer


class IRDRLHFTrainer:
    """
    Trainer combining IRD trajectory selection with RLHF.
    
    Pipeline:
    1. Generate candidate trajectories
    2. Select high-uncertainty pairs using IRD/ensemble/random
    3. Collect human preferences
    4. Update reward model
    5. (Optional) Train policy on learned reward
    """
    
    def __init__(
        self,
        world_size=(12, 12),
        state_dim=7,
        feature_dim=5,
        hidden_dim=64,
        uncertainty_method='ird',  # 'ird', 'ensemble', or 'random'
        num_posterior_samples=100
    ):
        self.world_size = world_size
        self.state_dim = state_dim
        self.uncertainty_method = uncertainty_method
        
        # Environment parameters
        self.env_kwargs = {
            'size': world_size,
            'num_seeds': 5,
            'num_rocks': 15
        }
        
        # Initialize world (will be reset for each trajectory)
        self.world = None
        
        # Initialize policy (used after reward learning)
        self.policy = PolicyNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=4
        )
        
        # Initialize reward model based on method
        if uncertainty_method == 'ird':
            self.reward_model = BayesianRewardModel(
                feature_dim=feature_dim,
                num_samples=num_posterior_samples,
                prior_std=1.0
            )
            # Set proxy reward weights
            # [seeds, clouds, battery_depletion, movement, final_battery]
            proxy_weights = np.array([10.0, -1.0, -5.0, -0.5, 2.0])
            self.reward_model.set_proxy_reward(proxy_weights)
            self.preference_learner = None  # IRD doesn't need neural learner
            
        elif uncertainty_method == 'ensemble':
            self.reward_model = NeuralRewardEnsemble(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_models=5
            )
            self.preference_learner = EnsemblePreferenceLearner(
                self.reward_model,
                learning_rate=1e-3
            )
            
        else:  # random
            self.reward_model = NeuralRewardModel(
                state_dim=state_dim,
                hidden_dim=hidden_dim
            )
            self.preference_learner = PreferenceLearner(
                self.reward_model,
                learning_rate=1e-3
            )
        
        # Preference dataset
        self.preference_dataset = PreferenceDataset()
        
        # Tracking
        self.uncertainty_history = []
        self.correlation_history = []
        
        print(f"IRD-RLHF Trainer initialized!")
        print(f"  Uncertainty method: {uncertainty_method}")
        print(f"  World size: {world_size}")
    
    def create_world(self, seed=None):
        """Create a new world instance."""
        world = SciWrld(**self.env_kwargs, random_seed=seed)
        
        # Add some clouds
        for _ in range(3):
            world.step(new_cloud_rate=0.5, cloud_limit=3, cloud_size=4)
        
        return world
    
    def generate_trajectory_candidates(self, num_candidates=20, steps=8):
        """
        Generate trajectory candidates with their world states.
        
        Each trajectory is generated in its own world instance to ensure
        proper state tracking.
        
        @return: list of (trajectory, world) tuples
        """
        candidates = []
        
        for i in range(num_candidates):
            # Create fresh world
            world = self.create_world(seed=np.random.randint(0, 100000))
            world.add_agent(Agent)
            
            # Generate random trajectory
            traj = world.agent.generate_trajectory(
                steps=steps,
                seed=np.random.randint(0, 100000)
            )
            
            candidates.append((traj, world))
        
        return candidates
    
    def compute_trajectory_uncertainty(self, trajectory, world):
        """
        Compute uncertainty for a trajectory.
        
        @return: (mean_reward, uncertainty)
        """
        if self.uncertainty_method == 'ird':
            return self.reward_model.compute_reward_uncertainty(world, trajectory)
            
        elif self.uncertainty_method == 'ensemble':
            states = encode_trajectory_states(world, trajectory)
            return self.reward_model.predict_trajectory_reward(states)
            
        else:  # random
            return 0.0, np.random.rand()
    
    def select_high_uncertainty_pairs(
        self,
        num_pairs=10,
        candidates_per_pair=20,
        steps=8,
        min_diversity=0.5
    ):
        """
        Select trajectory pairs with highest uncertainty for labeling.
        
        @param num_pairs: number of pairs to select
        @param candidates_per_pair: candidates to generate per selection
        @param steps: trajectory length
        @param min_diversity: minimum trajectory difference (0-1)
        @return: list of (traj1, world1, traj2, world2, unc1, unc2) tuples
        """
        selected_pairs = []
        
        print(f"\nSelecting {num_pairs} high-uncertainty trajectory pairs...")
        print(f"Method: {self.uncertainty_method}")
        
        for pair_idx in range(num_pairs):
            # Generate candidates
            candidates = self.generate_trajectory_candidates(
                num_candidates=candidates_per_pair,
                steps=steps
            )
            
            # Compute uncertainty for each
            uncertainties = []
            for traj, world in candidates:
                _, unc = self.compute_trajectory_uncertainty(traj, world)
                uncertainties.append(unc)
            
            uncertainties = np.array(uncertainties)
            
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort(uncertainties)[::-1]
            
            # Find diverse pair with high uncertainty
            best_pair = None
            best_unc_sum = -1
            
            for i, idx1 in enumerate(sorted_indices[:10]):
                for idx2 in sorted_indices[i+1:15]:
                    traj1, _ = candidates[idx1]
                    traj2, _ = candidates[idx2]
                    
                    # Check diversity
                    sim = trajectory_similarity(traj1, traj2)
                    if sim <= (1 - min_diversity):
                        unc_sum = uncertainties[idx1] + uncertainties[idx2]
                        if unc_sum > best_unc_sum:
                            best_unc_sum = unc_sum
                            best_pair = (idx1, idx2)
            
            # Fallback: just take top 2 if no diverse pair found
            if best_pair is None:
                best_pair = (sorted_indices[0], sorted_indices[1])
            
            idx1, idx2 = best_pair
            traj1, world1 = candidates[idx1]
            traj2, world2 = candidates[idx2]
            unc1, unc2 = uncertainties[idx1], uncertainties[idx2]
            
            selected_pairs.append((traj1, world1, traj2, world2, unc1, unc2))
            
            mean_unc = (unc1 + unc2) / 2
            self.uncertainty_history.append(mean_unc)
            
            if (pair_idx + 1) % 5 == 0:
                print(f"  Selected {pair_idx+1}/{num_pairs} pairs, avg unc: {mean_unc:.4f}")
        
        return selected_pairs
    
    def collect_preferences(
        self,
        num_preferences=50, # number of preferences to collect
        candidates_per_pair=20, # candidates per selection
        steps=8, # trajectory length
        simulated=False # if True, use true reward instead of human input
    ):

        print(f"\n{'='*60}")
        print("COLLECTING PREFERENCES")
        print(f"Method: {self.uncertainty_method}")
        print(f"Simulated: {simulated}")
        print(f"{'='*60}\n")
        
        # Select high-uncertainty pairs
        selected_pairs = self.select_high_uncertainty_pairs(
            num_pairs=num_preferences,
            candidates_per_pair=candidates_per_pair,
            steps=steps
        )
        
        # Collect preferences
        for i, (traj1, world1, traj2, world2, unc1, unc2) in enumerate(selected_pairs):
            print(f"\n--- Preference {i+1}/{num_preferences} ---")
            print(f"Uncertainty: traj1={unc1:.4f}, traj2={unc2:.4f}")
            
            if simulated:
                # Use true reward as oracle
                true_rew1 = compute_true_reward(world1, traj1)
                true_rew2 = compute_true_reward(world2, traj2)
                preferred_idx = 0 if true_rew1 >= true_rew2 else 1
                print(f"[Simulated] True rewards: {true_rew1:.2f} vs {true_rew2:.2f}")
                print(f"[Simulated] Preferred: trajectory {preferred_idx + 1}")
            else:
                # Get actual human preference
                preferred_idx = self._show_and_get_preference(
                    traj1, world1, traj2, world2
                )
            
            # Encode trajectories
            states1 = encode_trajectory_states(world1, traj1)
            states2 = encode_trajectory_states(world2, traj2)
            
            # Add to dataset
            self.preference_dataset.add_preference(states1, states2, preferred_idx)
            
            # Update IRD posterior if using that method
            if self.uncertainty_method == 'ird':
                self.reward_model.update_posterior_pairwise(
                    traj1, world1, traj2, world2, preferred_idx
                )        
                
        print(f"\nCollected {num_preferences} preferences!")
        
        # Train neural reward model if using ensemble or random
        if self.preference_learner is not None and len(self.preference_dataset) > 0:
            print("\nTraining reward model from preferences...")
            self.preference_learner.train(
                self.preference_dataset,
                epochs=50,
                batch_size=16
            )
    
    def _show_and_get_preference(self, traj1, world1, traj2, world2):

        print("\n--- TRAJECTORY 1 ---")
        print(world1.render(trajectory=traj1))
        true_rew1 = compute_true_reward(world1, traj1)
        print(f"[Hidden] True reward: {true_rew1:.2f}")
        
        print("\n--- TRAJECTORY 2 ---")
        print(world2.render(trajectory=traj2))
        true_rew2 = compute_true_reward(world2, traj2)
        print(f"[Hidden] True reward: {true_rew2:.2f}")
        
        while True:
            try:
                choice = int(input("\nWhich trajectory is better? (1 or 2): "))
                if choice in [1, 2]:
                    return choice - 1
                print("Please enter 1 or 2")
            except (ValueError, KeyboardInterrupt):
                print("Please enter 1 or 2")
    
    def evaluate_sample_efficiency(self, num_test=20):

        print(f"\n{'='*60}")
        print("EVALUATING SAMPLE EFFICIENCY")
        print(f"{'='*60}\n")
        
        # Generate test trajectories
        test_candidates = self.generate_trajectory_candidates(
            num_candidates=num_test,
            steps=8
        )
        
        true_rewards = []
        learned_rewards = []
        
        for traj, world in test_candidates:
            # True reward
            true_rew = compute_true_reward(world, traj)
            true_rewards.append(true_rew)
            
            # Learned reward
            if self.uncertainty_method == 'ird':
                mean_rew, _ = self.reward_model.compute_reward_uncertainty(world, traj)
            elif self.uncertainty_method == 'ensemble':
                states = encode_trajectory_states(world, traj)
                mean_rew, _ = self.reward_model.predict_trajectory_reward(states)
            else:
                states = encode_trajectory_states(world, traj)
                self.reward_model.eval()
                with torch.no_grad():
                    mean_rew = self.reward_model.predict_trajectory_reward(
                        torch.tensor(states, dtype=torch.float32)
                    ).item()
            
            learned_rewards.append(mean_rew)
        
        # Compute correlation
        corr = np.corrcoef(true_rewards, learned_rewards)[0, 1]
        self.correlation_history.append(corr)
        
        print(f"Correlation with true reward: {corr:.4f}")
        print(f"Preferences collected: {len(self.preference_dataset)}")
        
        return {
            'correlation': corr,
            'num_preferences': len(self.preference_dataset),
            'true_rewards': true_rewards,
            'learned_rewards': learned_rewards
        }
    
    def train_policy(self, num_episodes=500, max_steps=20):
        """
        Train policy to maximize learned reward.
        """
        trainer = PolicyTrainer(
            policy=self.policy,
            reward_model=self.reward_model,
            learning_rate=3e-4,
            gamma=0.99
        )
        
        trainer.train(
            env_class=SciWrld,
            env_kwargs=self.env_kwargs,
            agent_class=PolicyAgent,
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        
        return trainer
    
    def plot_results(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Uncertainty over time
        if self.uncertainty_history:
            ax1.plot(self.uncertainty_history)
            ax1.set_xlabel('Preference Pair')
            ax1.set_ylabel('Average Uncertainty')
            ax1.set_title(f'Uncertainty ({self.uncertainty_method})')
            ax1.grid(True, alpha=0.3)
        
        # Correlation over time
        if self.correlation_history:
            ax2.plot(self.correlation_history, marker='o')
            ax2.set_xlabel('Evaluation')
            ax2.set_ylabel('Correlation with True Reward')
            ax2.set_title(f'Learning Progress ({self.uncertainty_method})')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def compare_methods(num_preferences=30, simulated=True):

    methods = ['ird', 'ensemble', 'random']
    results = {}
    
    print("\n" + "="*60)
    print("COMPARING TRAJECTORY SELECTION METHODS")
    print("="*60)
    
    for method in methods:
        print(f"\n\n{'='*60}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*60}")
        
        trainer = IRDRLHFTrainer(
            world_size=(12, 12),
            uncertainty_method=method
        )
        
        # Collect preferences
        trainer.collect_preferences(
            num_preferences=num_preferences,
            candidates_per_pair=20,
            steps=8,
            simulated=simulated
        )
        
        # Evaluate
        eval_results = trainer.evaluate_sample_efficiency(num_test=30)
        results[method] = eval_results
        
        # Plot
        trainer.plot_results(save_path=f'results_{method}.png')
    
    # Print comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for method, res in results.items():
        print(f"{method:10s}: correlation = {res['correlation']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='IRD-enhanced RLHF training')
    parser.add_argument('--method', type=str, default='ird',
                       choices=['ird', 'ensemble', 'random'])
    parser.add_argument('--num_preferences', type=int, default=30)
    parser.add_argument('--candidates_per_pair', type=int, default=20)
    parser.add_argument('--simulated', action='store_true',
                       help='Use simulated preferences (true reward as oracle)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all methods')
    parser.add_argument('--train_policy', action='store_true',
                       help='Train policy after reward learning')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_methods(
            num_preferences=args.num_preferences,
            simulated=args.simulated
        )
    else:
        # Single method training
        trainer = IRDRLHFTrainer(
            world_size=(12, 12),
            uncertainty_method=args.method
        )
        
        # Collect preferences
        trainer.collect_preferences(
            num_preferences=args.num_preferences,
            candidates_per_pair=args.candidates_per_pair,
            steps=8,
            simulated=args.simulated
        )
        
        # Evaluate
        trainer.evaluate_sample_efficiency(num_test=30)
        
        # Plot
        trainer.plot_results(save_path=f'outputs/results_{args.method}.png')
        
        # Optionally train policy
        if args.train_policy:
            trainer.train_policy(num_episodes=500, max_steps=20)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    main()
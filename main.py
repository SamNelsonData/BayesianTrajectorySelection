"""
Training script combining Inverse Reward Design with RLHF.

Key innovation: Use Bayesian posterior over reward functions to select
trajectory segments with high uncertainty for human labeling.

This reduces the number of human labels needed compared to:
1. Random selection (baseline)
2. Ensemble variance (Christiano et al. 2017)

Usage:
    python train_ird.py --method ird --num_preferences 50
"""

import torch
import numpy as np
from arch.sciwrld import SciWrld, encode_state
from arch.agents import Agent, AgentA2C
from arch.policy import PolicyNetwork
from arch.reward_function import (
    BayesianRewardModel,
    NeuralRewardEnsemble,
    proxy_reward_function,
    true_reward_function,
    encode_trajectory_states
)
from arch.preference_learning import PreferenceDataset, PreferenceLearner
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt


class IRDRLHFTrainer:
    """
    Trainer that combines:
    1. Inverse Reward Design (IRD) for trajectory selection
    2. Human preferences for reward learning
    3. Policy optimization using learned rewards
    """
    
    def __init__(
        self,
        world_size=(12, 12),
        feature_dim=5,
        state_dim=7,
        hidden_dim=64,
        uncertainty_method='ird',  # 'ird', 'ensemble', or 'random'
        num_posterior_samples=100
    ):
        """
        @param uncertainty_method: how to select trajectories
            - 'ird': Bayesian IRD posterior variance
            - 'ensemble': Neural ensemble variance (Christiano et al.)
            - 'random': Random selection (baseline)
        """
        self.world_size = world_size
        self.uncertainty_method = uncertainty_method
        
        # Initialize environment
        self.world = SciWrld(size=world_size, starting_seeds=5, rocks=15)
        
        # Initialize policy
        self.policy = PolicyNetwork(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=4
        )
        
        # Initialize reward uncertainty model based on method
        if uncertainty_method == 'ird':
            self.reward_model = BayesianRewardModel(
                feature_dim=feature_dim,
                num_samples=num_posterior_samples,
                prior_std=1.0
            )
            # Set proxy reward (what designer gave to agent)
            proxy_weights = np.array([10.0, -1.0, -5.0, -0.5, 2.0])
            self.reward_model.set_proxy_reward(proxy_weights)
            
        elif uncertainty_method == 'ensemble':
            self.reward_model = NeuralRewardEnsemble(
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                num_models=5
            )
            # Will need to train ensemble from preferences
            self.preference_learner = None  # TODO: add ensemble training
            
        else:  # random
            self.reward_model = None
        
        # Dataset for human preferences
        self.preference_dataset = PreferenceDataset()
        
        # Tracking
        self.uncertainty_history = []
        self.correlation_history = []
        
        print(f"IRD-RLHF Trainer initialized!")
        print(f"Uncertainty method: {uncertainty_method}")
        print(f"World size: {world_size}")
    
    def reset_world(self, rng_seed=None):
        """Create fresh world instance with clouds and seeds."""
        if rng_seed is not None:
            np.random.seed(rng_seed)
        self.world = SciWrld(
            size=self.world_size,
            starting_seeds=5,
            rocks=15
        )
        
        # Add some clouds for variety
        for _ in range(2):
            self.world.step(
                steps=0,  # Don't move clouds yet
                new_cloud_rate=1.0,  # Force cloud generation
                cloud_limit=3,
                cloud_size=4
            )
        
        self.world.add_agent(Agent)
    
    def generate_trajectory_candidates(self, num_candidates=20, steps=8):
        """
        Generate multiple trajectory candidates for selection.
        NOW RETURNS: (trajectory, world_snapshot) pairs
        
        @param num_candidates: number of trajectories to generate
        @param steps: trajectory length
        @return: list of (trajectory, world) tuples
        """
        candidates = []
        
        for i in range(num_candidates):
            self.reset_world()
            
            # Generate random trajectory
            seed_val = np.random.randint(0, 100000)
            traj = self.world.agent.gen_trajectory(
                seeded=seed_val,
                steps=steps
            )
            
            # Store trajectory WITH its world state
            world_snapshot = deepcopy(self.world)
            candidates.append((traj, world_snapshot))
        
        return candidates
    
    def compute_trajectory_uncertainty(self, trajectory, world):
        """
        Compute uncertainty for a trajectory using selected method.
        
        @param trajectory: list of positions
        @param world: the world state where this trajectory exists
        @return: (mean_reward, uncertainty)
        """
        if self.uncertainty_method == 'ird':
            mean_reward, variance = self.reward_model.compute_reward_uncertainty(
                world, trajectory
            )
            return mean_reward, variance
            
        elif self.uncertainty_method == 'ensemble':
            trajectory_states = encode_trajectory_states(world, trajectory)
            mean_reward, variance = self.reward_model.predict_trajectory_reward(
                trajectory_states
            )
            return mean_reward, variance
            
        else:  # random
            # Return random uncertainty
            return 0.0, np.random.rand()
    
    def select_high_uncertainty_pairs(self, num_pairs=10, candidates_per_pair=20, steps=8):
        """
        Select trajectory pairs with highest uncertainty for labeling.
        
        This is the key innovation: instead of random pairs, select pairs
        where the reward model is most uncertain.
        
        @param num_pairs: number of trajectory pairs to select
        @param candidates_per_pair: number of candidates to consider per pair
        @param steps: trajectory length
        @return: list of (traj1, world1, traj2, world2, unc1, unc2) tuples
        """
        selected_pairs = []
        
        print(f"\nGenerating and selecting high-uncertainty trajectory pairs...")
        print(f"Method: {self.uncertainty_method}")
        
        for pair_idx in range(num_pairs):
            # Generate candidate trajectories WITH their world states
            candidates = self.generate_trajectory_candidates(
                num_candidates=candidates_per_pair,
                steps=steps
            )
            
            # Compute uncertainty for each candidate
            uncertainties = []
            for traj, world in candidates:
                _, uncertainty = self.compute_trajectory_uncertainty(traj, world)
                uncertainties.append(uncertainty)
            
            uncertainties = np.array(uncertainties)
            
            # Select top 2 with highest uncertainty
            top_indices = np.argsort(uncertainties)[-2:]
            
            traj1, world1 = candidates[top_indices[0]]
            traj2, world2 = candidates[top_indices[1]]
            unc1 = uncertainties[top_indices[0]]
            unc2 = uncertainties[top_indices[1]]
            
            selected_pairs.append((traj1, world1, traj2, world2, unc1, unc2))
            
            mean_unc = (unc1 + unc2) / 2
            self.uncertainty_history.append(mean_unc)
            
            if (pair_idx + 1) % 5 == 0:
                print(f"  Selected {pair_idx + 1}/{num_pairs} pairs, "
                      f"avg uncertainty: {mean_unc:.4f}")
        
        return selected_pairs
    
    def collect_preferences_with_uncertainty(
        self,
        num_preferences=50,
        candidates_per_pair=20,
        steps=8
    ):
        """
        Collect human preferences, prioritizing high-uncertainty trajectories.
        
        @param num_preferences: number of preferences to collect
        @param candidates_per_pair: candidates to generate per selection
        @param steps: trajectory length
        """
        print(f"\n{'='*60}")
        print(f"COLLECTING PREFERENCES WITH UNCERTAINTY SELECTION")
        print(f"Method: {self.uncertainty_method}")
        print(f"{'='*60}\n")
        
        # Select high-uncertainty pairs
        selected_pairs = self.select_high_uncertainty_pairs(
            num_pairs=num_preferences,
            candidates_per_pair=candidates_per_pair,
            steps=steps
        )
        
        # Collect human preferences for each pair
        for i, (traj1, world1, traj2, world2, unc1, unc2) in enumerate(selected_pairs):
            print(f"\n--- Preference {i+1}/{num_preferences} ---")
            print(f"Trajectory 1 uncertainty: {unc1:.4f}")
            print(f"Trajectory 2 uncertainty: {unc2:.4f}")
            
            # Get human preference (NOW PASSING THE CORRECT WORLDS!)
            preferred_idx = self.show_and_get_preference(traj1, world1, traj2, world2)
            
            # Encode trajectories with their respective worlds
            traj1_states = encode_trajectory_states(world1, traj1)
            traj2_states = encode_trajectory_states(world2, traj2)
            
            # Add to dataset
            self.preference_dataset.add_preference(
                traj1_states,
                traj2_states,
                preferred_idx
            )
            
            print(f"✓ Preference recorded: Trajectory {preferred_idx + 1}")
            
            # Update IRD posterior if using that method
            if self.uncertainty_method == 'ird':
                if preferred_idx == 0:
                    self.reward_model.update_posterior([traj1], world1, temperature=1.0)
                else:
                    self.reward_model.update_posterior([traj2], world2, temperature=1.0)
        
        print(f"\n✓ Collected {num_preferences} preferences!")
    
    def show_and_get_preference(self, traj1, world1, traj2, world2):
        """
        Display trajectories and get human preference.
        Uses CORRECT world states for each trajectory.
        
        @param traj1: first trajectory
        @param world1: world state for trajectory 1
        @param traj2: second trajectory  
        @param world2: world state for trajectory 2
        @return: preferred index (0 or 1)
        """
        
        
        # print("\n--- TRAJECTORY 1 ---")
        true_rew1 = self._simulate_and_display_trajectory(world1, traj1)

        # print("\n--- TRAJECTORY 2 ---")
        true_rew2 = self._simulate_and_display_trajectory(world2, traj2)
    
        # print(f"\n[Hidden] True rewards: Traj1={true_rew1:.2f}, Traj2={true_rew2:.2f}")
        
        # from utils.visualiser import TrajectoryVisualizer
        
        print(f"\n[Hidden] True rewards: Traj1={true_rew1:.2f}, Traj2={true_rew2:.2f}")
        
        # Show beautiful visualization
        # viz = TrajectoryVisualizer()
        # viz.show_trajectory_pair(
        #     world1, traj1, world2, traj2,
        #     reward1=None,  # Don't show reward to human
        #     reward2=None,
        #     title1="Trajectory 1",
        #     title2="Trajectory 2"
        # )
        
        while True:
            try:
                choice = int(input("\nWhich trajectory is better? (1 or 2): "))
                if choice in [1, 2]:
                    return choice - 1
                print("Please enter 1 or 2")
            except (ValueError, KeyboardInterrupt):
                print("Please enter 1 or 2")
    
    def _simulate_and_display_trajectory(self, world, trajectory):
        """
        Simulate and display a trajectory with the world state.
        Shows the gridworld with trajectory path overlaid.
        
        @param world: SciWrld instance
        @param trajectory: list of positions
        @return: total true reward
        """
        # Visualize the trajectory on the world
        self._visualize_trajectory_on_world(world, trajectory)
        
        # Simulate the trajectory and compute true reward
        total_reward = 0.0
        seeds_collected = 0
        time_under_clouds = 0
        battery_warnings = 0
        
        world.agent.position = trajectory[0]
        world.agent.battery = 2  # Reset battery
        
        for i, pos in enumerate(trajectory):
            world.agent.position = pos
            
            # Living penalty
            step_reward = -0.1
            
            # Check for seed collection BEFORE moving there
            if world.world[pos] == world.item_to_value['Seed']:
                seeds_collected += 1
                step_reward += 10.0  # Seed reward
                world.world[pos] = world.item_to_value['Sand']  # Remove seed
            
            # Check for clouds
            under_cloud = False
            for cloud, _ in world.clouds:
                if pos in cloud:
                    under_cloud = True
                    time_under_clouds += 1
                    world.agent.battery -= 1
                    step_reward -= 5.0  # Cloud penalty
                    break
            
            # Recharge in sunlight
            if not under_cloud and world.agent.battery < 2:
                world.agent.battery += 1
            
            # Battery depletion penalty
            if world.agent.battery <= 0:
                battery_warnings += 1
                step_reward -= 20.0
            
            total_reward += step_reward
        
        # Print summary
        print(f"Seeds collected: {seeds_collected}")
        print(f"Time under clouds: {time_under_clouds}")
        print(f"Battery warnings: {battery_warnings}")
        print(f"Total reward: {total_reward:.2f}")
        
        return total_reward
    
    def _visualize_trajectory_on_world(self, world, trajectory):
        """
        Display the world with trajectory path overlaid.
        """
        # Create a visual representation
        direction_symbols = {
            (0, -1): '←',
            (0, 1): '→',
            (-1, 0): '↑',
            (1, 0): '↓',
            (0, 0): '•'
        }
        
        # Build trajectory map
        traj_map = {}
        for i in range(len(trajectory) - 1):
            pos = trajectory[i]
            next_pos = trajectory[i + 1]
            direction = (next_pos[0] - pos[0], next_pos[1] - pos[1])
            traj_map[pos] = direction_symbols.get(direction, '•')
        
        # Mark end position
        if trajectory:
            traj_map[trajectory[-1]] = '⊗'
        
        # Print the world
        for r in range(world.size[0]):
            row_str = ""
            for c in range(world.size[1]):
                pos = (r, c)
                
                # Check if there's a trajectory marker here
                if pos in traj_map:
                    symbol = traj_map[pos]
                else:
                    # Show world element
                    val = world.world[pos]
                    symbol = world.value_to_symbol[val]
                    
                    # Override with cloud if present
                    for cloud, _ in world.clouds:
                        if pos in cloud:
                            symbol = '⛆'
                            break
                
                row_str += f'{symbol:<2}'
            print(row_str)
        print()
    
    def evaluate_sample_efficiency(self, num_trials=5):
        """
        Compare sample efficiency across uncertainty methods.
        
        Measures how quickly learned reward correlates with true reward
        as a function of number of labeled pairs.
        
        @param num_trials: number of evaluation trials
        @return: dict of results
        """
        print(f"\n{'='*60}")
        print("EVALUATING SAMPLE EFFICIENCY")
        print(f"{'='*60}\n")
        
        correlations = []
        
        for trial in range(num_trials):
            # Generate test trajectories
            test_trajs = self.generate_trajectory_candidates(
                num_candidates=20,
                steps=8
            )
            
            true_rewards = []
            learned_rewards = []
            
            for traj in test_trajs:
                # True reward
                true_rew = sum(true_reward_function(self.world, pos) for pos in traj[0])      
                true_rewards.append(true_rew)
                
                # Learned reward (mean of posterior)
                if self.uncertainty_method == 'ird':
                    mean_rew, _ = self.reward_model.compute_reward_uncertainty(
                        self.world, traj[0]
                    )
                    learned_rewards.append(mean_rew)
                # TODO: Add ensemble evaluation
            
            # Compute correlation
            if len(true_rewards) > 1:
                corr = np.corrcoef(true_rewards, learned_rewards)[0, 1]
                correlations.append(corr)
                self.correlation_history.append(corr)
        
        avg_corr = np.mean(correlations)
        print(f"Average correlation with true reward: {avg_corr:.4f}")
        
        return {
            'correlations': correlations,
            'avg_correlation': avg_corr,
            'method': self.uncertainty_method
        }
    
    def plot_results(self):
        """Plot uncertainty and correlation over time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot uncertainty over time
        ax1.plot(self.uncertainty_history)
        ax1.set_xlabel('Preference Pair')
        ax1.set_ylabel('Average Uncertainty')
        ax1.set_title(f'Uncertainty Over Time ({self.uncertainty_method})')
        ax1.grid(True)
        
        # Plot correlation over time
        ax2.plot(self.correlation_history)
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Correlation with True Reward')
        ax2.set_title(f'Learning Progress ({self.uncertainty_method})')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'ird_rlhf_results_{self.uncertainty_method}.png')
        print(f"\n✓ Results saved to ird_rlhf_results_{self.uncertainty_method}.png")


def compare_methods(num_preferences_per_method=30):
    """
    Compare IRD vs Ensemble vs Random trajectory selection.
    
    @param num_preferences_per_method: labels to collect per method
    """
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
        trainer.collect_preferences_with_uncertainty(
            num_preferences=num_preferences_per_method,
            candidates_per_pair=20,
            steps=8
        )
        
        # Evaluate
        eval_results = trainer.evaluate_sample_efficiency(num_trials=5)
        results[method] = eval_results
        
        # Plot
        trainer.plot_results()
    
    # Compare results
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for method, res in results.items():
        print(f"{method:10s}: correlation = {res['avg_correlation']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='IRD-enhanced RLHF training'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='ird',
        choices=['ird', 'ensemble', 'random'],
        help='Trajectory selection method'
    )
    parser.add_argument(
        '--num_preferences',
        type=int,
        default=50,
        help='Number of preferences to collect'
    )
    parser.add_argument(
        '--candidates_per_pair',
        type=int,
        default=20,
        help='Candidate trajectories to generate per pair'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all methods'
    )
    
    args = parser.parse_args()
    
    # TODO comment bottom two lines
    if args.compare:
        compare_methods(num_preferences_per_method=args.num_preferences)
    else:
        # Single method training
        trainer = IRDRLHFTrainer(
            world_size=(12, 12),
            uncertainty_method=args.method
        )
                
        # Collect preferences with uncertainty-based selection
        trainer.collect_preferences_with_uncertainty(
            num_preferences=args.num_preferences,
            candidates_per_pair=args.candidates_per_pair,
            steps=8
        )
        
        # Evaluate
        trainer.evaluate_sample_efficiency(num_trials=5)
        
        # Plot results
        trainer.plot_results()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Method: {args.method}")
        print(f"Preferences collected: {args.num_preferences}")
        print(f"\nNext steps:")
        print("  1. Train policy using learned reward")
        print("  2. Iterate: collect more preferences with improved policy")
        print("  3. Compare sample efficiency across methods")
        
        # from test import RewardModelTester

        # # Assuming you have a trained trainer
        # tester = RewardModelTester(trainer)
        # tester.run_all_tests(num_test=50)


if __name__ == "__main__":
    main()
"""
Test script for evaluating trained reward models.

Tests:
1. Correlation with ground truth rewards
2. Ranking accuracy (can it order trajectories correctly?)
3. Feature weight analysis (what did it learn?)
4. Uncertainty calibration (is high uncertainty meaningful?)

Usage:
    python test_model.py --method ird --num_test 50
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import argparse
from main import IRDRLHFTrainer
from arch.sciwrld import SciWrld
from arch.agents import Agent
from copy import deepcopy


class RewardModelTester:
    def __init__(self, trainer):

        self.trainer = trainer
        self.results = {}
    
    def test_correlation(self, num_trajectories=50):
        """
        Test 1: Correlation between learned and true rewards.
        
        Measures how well the learned reward function correlates
        with ground truth on unseen trajectories.
        
        @param num_trajectories: number of test trajectories
        @return: dict with Pearson and Spearman correlations
        """
        print(f"\n{'='*60}")
        print("TEST 1: REWARD CORRELATION")
        print(f"{'='*60}\n")
        
        true_rewards = []
        learned_rewards = []
        uncertainties = []
        
        for i in range(num_trajectories):
            # Generate test trajectory
            self.trainer.reset_world()
            traj = self.trainer.world.agent.gen_trajectory(steps=8)
            world_snapshot = deepcopy(self.trainer.world)
            
            # Compute TRUE reward (ground truth)
            true_rew = self.trainer._simulate_and_display_trajectory(
                world_snapshot, traj
            ) if hasattr(self.trainer, '_simulate_and_display_trajectory') else 0
            
            # Compute LEARNED reward
            if self.trainer.uncertainty_method == 'ird':
                mean_rew, unc = self.trainer.reward_model.compute_reward_uncertainty(
                    world_snapshot, traj
                )
            else:
                mean_rew, unc = 0.0, 0.0
            
            true_rewards.append(true_rew)
            learned_rewards.append(mean_rew)
            uncertainties.append(unc)
            
            if i < 5:  # Show first 5 examples
                print(f"Trajectory {i+1}:")
                print(f"  True reward: {true_rew:.2f}")
                print(f"  Learned reward: {mean_rew:.2f}")
                print(f"  Uncertainty: {unc:.4f}")
                print(f"  Error: {abs(true_rew - mean_rew):.2f}\n")
        
        # Compute correlations
        pearson = np.corrcoef(true_rewards, learned_rewards)[0, 1]
        spearman, _ = spearmanr(true_rewards, learned_rewards)
        
        print(f"Pearson correlation: {pearson:.4f}")
        print(f"Spearman correlation: {spearman:.4f}")
        print(f"Mean absolute error: {np.mean(np.abs(np.array(true_rewards) - np.array(learned_rewards))):.2f}")
        
        self.results['correlation'] = {
            'pearson': pearson,
            'spearman': spearman,
            'true_rewards': true_rewards,
            'learned_rewards': learned_rewards,
            'uncertainties': uncertainties
        }
        
        return self.results['correlation']
    
    def test_ranking_accuracy(self, num_pairs=30):
        """
        Test 2: Ranking accuracy.
        
        Can the model correctly rank which trajectory is better?
        This is what humans actually label!
        
        @param num_pairs: number of trajectory pairs to test
        @return: ranking accuracy (0-1)
        """
        print(f"\n{'='*60}")
        print("TEST 2: RANKING ACCURACY")
        print(f"{'='*60}\n")
        
        correct = 0
        total = 0
        
        for i in range(num_pairs):
            # Generate two trajectories
            candidates = self.trainer.generate_trajectory_candidates(
                num_candidates=2, steps=8
            )
            (traj1, world1), (traj2, world2) = candidates
            
            # TRUE ranking
            true_rew1 = self._compute_true_reward(world1, traj1)
            true_rew2 = self._compute_true_reward(world2, traj2)
            true_better = 0 if true_rew1 > true_rew2 else 1
            
            # LEARNED ranking
            if self.trainer.uncertainty_method == 'ird':
                learned_rew1, _ = self.trainer.reward_model.compute_reward_uncertainty(
                    world1, traj1
                )
                learned_rew2, _ = self.trainer.reward_model.compute_reward_uncertainty(
                    world2, traj2
                )
            else:
                learned_rew1, learned_rew2 = 0.0, 0.0
            
            learned_better = 0 if learned_rew1 > learned_rew2 else 1
            
            # Check if ranking matches
            if true_better == learned_better:
                correct += 1
            
            total += 1
            
            if i < 5:  # Show first 5 examples
                match = "✓" if true_better == learned_better else "✗"
                print(f"Pair {i+1}: {match}")
                print(f"  True:    Traj{true_better+1} better ({true_rew1:.2f} vs {true_rew2:.2f})")
                print(f"  Learned: Traj{learned_better+1} better ({learned_rew1:.2f} vs {learned_rew2:.2f})\n")
        
        accuracy = correct / total
        print(f"Ranking Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        self.results['ranking_accuracy'] = accuracy
        return accuracy
    
    def test_feature_weights(self):
        """
        Test 3: Analyze learned feature weights.
        
        What did the model actually learn about each feature?
        Compare to proxy weights and true reward structure.
        
        @return: mean weights and standard deviations
        """
        print(f"\n{'='*60}")
        print("TEST 3: LEARNED FEATURE WEIGHTS")
        print(f"{'='*60}\n")
        
        if self.trainer.uncertainty_method != 'ird':
            print("Feature weight analysis only available for IRD method")
            return None
        
        # Get posterior statistics
        mean_weights = self.trainer.reward_model.get_mean_weights()
        std_weights = self.trainer.reward_model.get_reward_std()
        proxy_weights = self.trainer.reward_model.proxy_weights
        
        feature_names = [
            "Seeds collected",
            "Time under clouds", 
            "Battery depletions",
            "Avg distance",
            "Final battery"
        ]
        
        print("Feature Weights:")
        print(f"{'Feature':<20} {'Proxy':<12} {'Learned':<12} {'Std':<12}")
        print("-" * 60)
        
        for i, name in enumerate(feature_names):
            print(f"{name:<20} {proxy_weights[i]:>10.2f}  {mean_weights[i]:>10.2f}  ±{std_weights[i]:>8.2f}")
        
        print("\nInterpretation:")
        if mean_weights[0] > 5:
            print("✓ Model learned to value seed collection")
        if mean_weights[1] < 0:
            print("✓ Model learned to avoid clouds")
        if mean_weights[2] < 0:
            print("✓ Model learned battery depletions are bad")
        
        self.results['feature_weights'] = {
            'mean': mean_weights,
            'std': std_weights,
            'proxy': proxy_weights,
            'names': feature_names
        }
        
        return self.results['feature_weights']
    
    def test_uncertainty_calibration(self, num_bins=5):
        """
        Test 4: Uncertainty calibration.
        
        Is high uncertainty actually associated with higher error?
        Good uncertainty means: uncertain predictions = less accurate.
        
        @param num_bins: number of uncertainty bins
        @return: calibration statistics
        """
        print(f"\n{'='*60}")
        print("TEST 4: UNCERTAINTY CALIBRATION")
        print(f"{'='*60}\n")
        
        if 'correlation' not in self.results:
            print("Run test_correlation() first!")
            return None
        
        true_rews = np.array(self.results['correlation']['true_rewards'])
        learned_rews = np.array(self.results['correlation']['learned_rewards'])
        uncertainties = np.array(self.results['correlation']['uncertainties'])
        
        errors = np.abs(true_rews - learned_rews)
        
        # Bin by uncertainty
        unc_percentiles = np.percentile(uncertainties, np.linspace(0, 100, num_bins + 1))
        
        print(f"{'Uncertainty Bin':<20} {'Mean Error':<15} {'Count'}")
        print("-" * 50)
        
        for i in range(num_bins):
            mask = (uncertainties >= unc_percentiles[i]) & (uncertainties < unc_percentiles[i+1])
            if mask.sum() > 0:
                mean_error = errors[mask].mean()
                count = mask.sum()
                print(f"[{unc_percentiles[i]:6.2f}, {unc_percentiles[i+1]:6.2f}]  {mean_error:>10.2f}       {count}")
        
        # Correlation between uncertainty and error
        unc_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\nCorrelation (uncertainty × error): {unc_error_corr:.4f}")
        
        if unc_error_corr > 0.3:
            print("✓ Good calibration: high uncertainty → high error")
        elif unc_error_corr > 0:
            print("~ Weak calibration: slight correlation")
        else:
            print("✗ Poor calibration: uncertainty not predictive of error")
        
        self.results['calibration'] = {
            'uncertainty_error_correlation': unc_error_corr
        }
        
        return self.results['calibration']
    
    def _compute_true_reward(self, world, trajectory):
        """Helper to compute true reward for a trajectory."""
        total_reward = 0.0
        world_copy = deepcopy(world)
        world_copy.agent.battery = 2
        
        for pos in trajectory:
            step_reward = -0.1
            
            if world_copy.world[pos] == world_copy.item_to_value['Seed']:
                step_reward += 10.0
                world_copy.world[pos] = world_copy.item_to_value['Sand']
            
            under_cloud = any(pos in cloud for cloud, _ in world_copy.clouds)
            if under_cloud:
                step_reward -= 5.0
                world_copy.agent.battery -= 1
            else:
                world_copy.agent.battery = min(2, world_copy.agent.battery + 1)
            
            if world_copy.agent.battery <= 0:
                step_reward -= 20.0
            
            total_reward += step_reward
        
        return total_reward
    
    def plot_results(self):
        """
        Create visualizations of test results.
        """
        if 'correlation' not in self.results:
            print("No results to plot! Run tests first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: True vs Learned Rewards
        ax1 = axes[0, 0]
        true_rews = self.results['correlation']['true_rewards']
        learned_rews = self.results['correlation']['learned_rewards']
        
        ax1.scatter(true_rews, learned_rews, alpha=0.6)
        ax1.plot([min(true_rews), max(true_rews)], 
                [min(true_rews), max(true_rews)], 
                'r--', label='Perfect correlation')
        ax1.set_xlabel('True Reward')
        ax1.set_ylabel('Learned Reward')
        ax1.set_title('Reward Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        ax2 = axes[0, 1]
        errors = np.abs(np.array(true_rews) - np.array(learned_rews))
        ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty vs Error
        ax3 = axes[1, 0]
        uncertainties = self.results['correlation']['uncertainties']
        ax3.scatter(uncertainties, errors, alpha=0.6)
        ax3.set_xlabel('Uncertainty (Variance)')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Uncertainty Calibration')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature Weights (if available)
        ax4 = axes[1, 1]
        if 'feature_weights' in self.results:
            weights = self.results['feature_weights']
            names = weights['names']
            mean_w = weights['mean']
            std_w = weights['std']
            proxy_w = weights['proxy']
            
            x = np.arange(len(names))
            width = 0.35
            
            ax4.bar(x - width/2, proxy_w, width, label='Proxy', alpha=0.7)
            ax4.bar(x + width/2, mean_w, width, label='Learned', alpha=0.7)
            ax4.errorbar(x + width/2, mean_w, yerr=std_w, fmt='none', color='black', capsize=5)
            
            ax4.set_xlabel('Feature')
            ax4.set_ylabel('Weight')
            ax4.set_title('Feature Weights: Proxy vs Learned')
            ax4.set_xticks(x)
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Feature weights\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(f'test_results_{self.trainer.uncertainty_method}.png', dpi=150)
        print(f"\n✓ Plots saved to test_results_{self.trainer.uncertainty_method}.png")
    
    def run_all_tests(self, num_test=50):
        """
        Run all tests and generate report.
        
        @param num_test: number of test trajectories
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL TESTING")
        print("="*60)
        
        # Run all tests
        self.test_correlation(num_trajectories=num_test)
        self.test_ranking_accuracy(num_pairs=num_test//2)
        self.test_feature_weights()
        self.test_uncertainty_calibration()
        
        # Generate visualizations
        self.plot_results()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Method: {self.trainer.uncertainty_method}")
        print(f"Preferences collected: {len(self.trainer.preference_dataset)}")
        if 'correlation' in self.results:
            print(f"Pearson correlation: {self.results['correlation']['pearson']:.4f}")
        if 'ranking_accuracy' in self.results:
            print(f"Ranking accuracy: {self.results['ranking_accuracy']:.2%}")
        
        print("\n✓ All tests complete!")

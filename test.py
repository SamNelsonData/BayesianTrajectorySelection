"""
Test script for evaluating trained reward models.

Tests:
1. Correlation with ground truth rewards
2. Ranking accuracy (can it order trajectories correctly?)
3. Feature weight analysis (IRD only)
4. Uncertainty calibration

Usage:
    python test_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from arch.sciwrld import SciWrld
from arch.agents import Agent
from arch.reward_function import compute_true_reward, encode_trajectory_states


class RewardModelTester:
    """
    Comprehensive testing for learned reward models.
    """
    
    def __init__(self, trainer):
        """
        @param trainer: IRDRLHFTrainer instance with trained reward model
        """
        self.trainer = trainer
        self.results = {}
    
    def test_correlation(self, num_trajectories=50):
        """
        Test correlation between learned and true rewards.
        
        @return: dict with Pearson and Spearman correlations
        """
        print(f"\n{'='*60}")
        print("TEST 1: REWARD CORRELATION")
        print(f"{'='*60}\n")
        
        true_rewards = []
        learned_rewards = []
        uncertainties = []
        
        # Generate test trajectories
        test_candidates = self.trainer.generate_trajectory_candidates(
            num_candidates=num_trajectories,
            steps=8
        )
        
        for i, (traj, world) in enumerate(test_candidates):
            # TRUE reward
            true_rew = compute_true_reward(world, traj)
            true_rewards.append(true_rew)
            
            # LEARNED reward and uncertainty
            mean_rew, unc = self.trainer.compute_trajectory_uncertainty(traj, world)
            learned_rewards.append(mean_rew)
            uncertainties.append(unc)
            
            if i < 5:
                print(f"Trajectory {i+1}:")
                print(f"  True: {true_rew:.2f}, Learned: {mean_rew:.2f}, Unc: {unc:.4f}")
                print(f"  Error: {abs(true_rew - mean_rew):.2f}\n")
        
        # Compute correlations
        pearson = np.corrcoef(true_rewards, learned_rewards)[0, 1]
        spearman, _ = spearmanr(true_rewards, learned_rewards)
        mae = np.mean(np.abs(np.array(true_rewards) - np.array(learned_rewards)))
        
        print(f"Pearson correlation: {pearson:.4f}")
        print(f"Spearman correlation: {spearman:.4f}")
        print(f"Mean absolute error: {mae:.2f}")
        
        self.results['correlation'] = {
            'pearson': pearson,
            'spearman': spearman,
            'mae': mae,
            'true_rewards': true_rewards,
            'learned_rewards': learned_rewards,
            'uncertainties': uncertainties
        }
        
        return self.results['correlation']
    
    def test_ranking_accuracy(self, num_pairs=30):
        """
        Test if model correctly ranks which trajectory is better.
        This is what humans actually label!
        
        @return: accuracy (0-1)
        """
        print(f"\n{'='*60}")
        print("TEST 2: RANKING ACCURACY")
        print(f"{'='*60}\n")
        
        correct = 0
        total = 0
        
        for i in range(num_pairs):
            # Generate two trajectories
            candidates = self.trainer.generate_trajectory_candidates(
                num_candidates=2,
                steps=8
            )
            (traj1, world1), (traj2, world2) = candidates
            
            # TRUE ranking
            true_rew1 = compute_true_reward(world1, traj1)
            true_rew2 = compute_true_reward(world2, traj2)
            true_better = 0 if true_rew1 > true_rew2 else 1
            
            # LEARNED ranking
            learned_rew1, _ = self.trainer.compute_trajectory_uncertainty(traj1, world1)
            learned_rew2, _ = self.trainer.compute_trajectory_uncertainty(traj2, world2)
            learned_better = 0 if learned_rew1 > learned_rew2 else 1
            
            if true_better == learned_better:
                correct += 1
            total += 1
            
            if i < 5:
                match = "✓" if true_better == learned_better else "✗"
                print(f"Pair {i+1}: {match}")
                print(f"  True:    Traj{true_better+1} ({true_rew1:.2f} vs {true_rew2:.2f})")
                print(f"  Learned: Traj{learned_better+1} ({learned_rew1:.2f} vs {learned_rew2:.2f})\n")
        
        accuracy = correct / max(total, 1)
        print(f"Ranking Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        self.results['ranking_accuracy'] = accuracy
        return accuracy
    
    def test_feature_weights(self):
        """
        Analyze learned feature weights (IRD only).
        """
        print(f"\n{'='*60}")
        print("TEST 3: FEATURE WEIGHTS (IRD)")
        print(f"{'='*60}\n")
        
        if self.trainer.uncertainty_method != 'ird':
            print("Feature weight analysis only available for IRD method")
            return None
        
        model = self.trainer.reward_model
        
        mean_weights = model.get_mean_weights()
        std_weights = model.get_std_weights()
        proxy_weights = model.proxy_weights
        
        print(f"{'Feature':<25} {'Proxy':<10} {'Learned':<10} {'Std':<10}")
        print("-" * 60)
        
        for i, name in enumerate(model.FEATURE_NAMES):
            print(f"{name:<25} {proxy_weights[i]:>8.2f}  {mean_weights[i]:>8.2f}  ±{std_weights[i]:<6.2f}")
        
        # Interpretation
        print("\nInterpretation:")
        if mean_weights[0] > 3:
            print("  ✓ Model values seed collection")
        if mean_weights[1] < -1:
            print("  ✓ Model learned to avoid clouds")
        if mean_weights[2] < -1:
            print("  ✓ Model penalizes battery depletion")
        
        self.results['feature_weights'] = {
            'mean': mean_weights,
            'std': std_weights,
            'proxy': proxy_weights,
            'names': model.FEATURE_NAMES
        }
        
        return self.results['feature_weights']
    
    def test_uncertainty_calibration(self, num_bins=5):
        """
        Test if high uncertainty correlates with high error.
        Good calibration: uncertain predictions should be less accurate.
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
        percentiles = np.percentile(uncertainties, np.linspace(0, 100, num_bins + 1))
        
        print(f"{'Uncertainty Range':<25} {'Mean Error':<15} {'Count'}")
        print("-" * 50)
        
        bin_errors = []
        for i in range(num_bins):
            mask = (uncertainties >= percentiles[i]) & (uncertainties < percentiles[i+1])
            if i == num_bins - 1:  # Include upper bound in last bin
                mask = (uncertainties >= percentiles[i]) & (uncertainties <= percentiles[i+1])
            
            if mask.sum() > 0:
                mean_error = errors[mask].mean()
                bin_errors.append(mean_error)
                print(f"[{percentiles[i]:6.2f}, {percentiles[i+1]:6.2f}]   {mean_error:>10.2f}       {mask.sum()}")
        
        # Correlation between uncertainty and error
        unc_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
        print(f"\nCorrelation (uncertainty × error): {unc_error_corr:.4f}")
        
        if unc_error_corr > 0.3:
            print("✓ Good calibration: high uncertainty → high error")
        elif unc_error_corr > 0:
            print("~ Weak calibration")
        else:
            print("✗ Poor calibration")
        
        self.results['calibration'] = {
            'unc_error_correlation': unc_error_corr,
            'bin_errors': bin_errors
        }
        
        return self.results['calibration']
    
    def plot_results(self, save_path=None):
        """Generate visualization of all test results."""
        if 'correlation' not in self.results:
            print("No results to plot! Run tests first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: True vs Learned
        ax1 = axes[0, 0]
        true = self.results['correlation']['true_rewards']
        learned = self.results['correlation']['learned_rewards']
        
        ax1.scatter(true, learned, alpha=0.6)
        lims = [min(min(true), min(learned)), max(max(true), max(learned))]
        ax1.plot(lims, lims, 'r--', label='Perfect')
        ax1.set_xlabel('True Reward')
        ax1.set_ylabel('Learned Reward')
        ax1.set_title(f"Correlation (r={self.results['correlation']['pearson']:.3f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        errors = np.abs(np.array(true) - np.array(learned))
        ax2.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(errors), color='r', linestyle='--',
                   label=f'Mean: {np.mean(errors):.2f}')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Uncertainty vs Error
        ax3 = axes[1, 0]
        uncertainties = self.results['correlation']['uncertainties']
        ax3.scatter(uncertainties, errors, alpha=0.6)
        ax3.set_xlabel('Uncertainty')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Uncertainty Calibration')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature weights (if available)
        ax4 = axes[1, 1]
        if 'feature_weights' in self.results:
            fw = self.results['feature_weights']
            x = np.arange(len(fw['names']))
            width = 0.35
            
            ax4.bar(x - width/2, fw['proxy'], width, label='Proxy', alpha=0.7)
            ax4.bar(x + width/2, fw['mean'], width, label='Learned', alpha=0.7)
            ax4.errorbar(x + width/2, fw['mean'], yerr=fw['std'],
                        fmt='none', color='black', capsize=5)
            
            ax4.set_ylabel('Weight')
            ax4.set_title('Feature Weights')
            ax4.set_xticks(x)
            ax4.set_xticklabels(fw['names'], rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Feature weights\nnot available',
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def run_all_tests(self, num_test=50, plot=True):
        """Run all tests and generate report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL TESTING")
        print(f"Method: {self.trainer.uncertainty_method}")
        print("="*60)
        
        self.test_correlation(num_trajectories=num_test)
        self.test_ranking_accuracy(num_pairs=num_test // 2)
        self.test_feature_weights()
        self.test_uncertainty_calibration()
        
        if plot:
            self.plot_results(
                save_path=f'outputs/test_results_{self.trainer.uncertainty_method}.png'
            )
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Method: {self.trainer.uncertainty_method}")
        print(f"Preferences: {len(self.trainer.preference_dataset)}")
        
        if 'correlation' in self.results:
            print(f"Pearson correlation: {self.results['correlation']['pearson']:.4f}")
            print(f"Spearman correlation: {self.results['correlation']['spearman']:.4f}")
        
        if 'ranking_accuracy' in self.results:
            print(f"Ranking accuracy: {self.results['ranking_accuracy']:.2%}")
        
        if 'calibration' in self.results:
            print(f"Uncertainty-error correlation: {self.results['calibration']['unc_error_correlation']:.4f}")
        
        print("\n✓ Testing complete!")
        
        return self.results


def main():
    """Demo: train a model and test it."""
    from main import IRDRLHFTrainer
    
    print("Training IRD model...")
    trainer = IRDRLHFTrainer(
        world_size=(12, 12),
        uncertainty_method='ird'
    )
    
    # Collect simulated preferences
    trainer.collect_preferences(
        num_preferences=30,
        simulated=True
    )
    
    # Test
    tester = RewardModelTester(trainer)
    tester.run_all_tests(num_test=50)


if __name__ == "__main__":
    main()
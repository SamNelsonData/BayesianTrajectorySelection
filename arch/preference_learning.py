import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple

class PreferenceDataset(Dataset):
    """
    Dataset for storing human preferences between trajectory pairs.
    Each sample is (traj1_states, traj2_states, preference)
    where preference is 0 if traj1 preferred, 1 if traj2 preferred.
    """
    
    def __init__(self):
        self.traj_pairs = []  # list of (traj1_states, traj2_states)
        self.preferences = []  # list of preferred indices (0 or 1)
    
    def add_preference(self, traj1_states, traj2_states, preferred_idx):
        """
        Add a preference comparison.
        
        @param traj1_states: encoded states for trajectory 1 [T, state_dim]
        @param traj2_states: encoded states for trajectory 2 [T, state_dim]
        @param preferred_idx: 0 if traj1 preferred, 1 if traj2 preferred
        """
        self.traj_pairs.append((traj1_states, traj2_states))
        self.preferences.append(preferred_idx)
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        traj1, traj2 = self.traj_pairs[idx]
        pref = self.preferences[idx]
        
        return (
            torch.tensor(traj1, dtype=torch.float32),
            torch.tensor(traj2, dtype=torch.float32),
            torch.tensor(pref, dtype=torch.long)
        )


class PreferenceLearner:
    """
    Trains a reward model from human preferences using the
    Bradley-Terry model: P(traj1 > traj2) = exp(R(traj1)) / (exp(R(traj1)) + exp(R(traj2)))
    """
    
    def __init__(self, reward_model, learning_rate=1e-3):
        """
        @param reward_model: RewardModel instance to train
        @param learning_rate: learning rate for optimizer
        """
        self.reward_model = reward_model
        self.optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def compute_preference_loss(self, traj1_states, traj2_states, preferred_idx):
        """
        Compute cross-entropy loss for preference prediction.
        
        Uses Bradley-Terry model:
        P(traj_i preferred) = exp(R(traj_i)) / (exp(R(traj1)) + exp(R(traj2)))
        
        @param traj1_states: states for trajectory 1 [batch, T, state_dim]
        @param traj2_states: states for trajectory 2 [batch, T, state_dim]
        @param preferred_idx: which trajectory preferred [batch]
        @return: loss value
        """
        # Compute trajectory rewards (sum over timesteps)
        batch_size = traj1_states.shape[0]
        
        r1_list = []
        r2_list = []
        
        for i in range(batch_size):
            r1 = self.reward_model.predict_trajectory_reward(traj1_states[i])
            r2 = self.reward_model.predict_trajectory_reward(traj2_states[i])
            r1_list.append(r1)
            r2_list.append(r2)
        
        r1 = torch.stack(r1_list)
        r2 = torch.stack(r2_list)
        
        # Bradley-Terry model: logits for softmax
        logits = torch.stack([r1, r2], dim=1).squeeze(-1)  # [batch, 2]
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits, preferred_idx)
        
        return loss
    
    def train_step(self, traj1_states, traj2_states, preferred_idx):
        """
        Single training step.
        
        @return: loss value
        """
        self.optimizer.zero_grad()
        loss = self.compute_preference_loss(traj1_states, traj2_states, preferred_idx)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, verbose=True):
        """
        Train for one epoch over the dataset.
        
        @param dataloader: DataLoader with PreferenceDataset
        @param verbose: whether to print progress
        @return: average loss
        """
        total_loss = 0.0
        num_batches = 0
        
        self.reward_model.train()
        
        for batch_idx, (traj1, traj2, pref) in enumerate(dataloader):
            loss = self.train_step(traj1, traj2, pref)
            total_loss += loss
            num_batches += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, dataset, epochs=50, batch_size=32, verbose=True):
        """
        Train the reward model on a preference dataset.
        
        @param dataset: PreferenceDataset instance
        @param epochs: number of training epochs
        @param batch_size: batch size
        @param verbose: whether to print progress
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset, skipping training")
            return
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training on {len(dataset)} preference pairs for {epochs} epochs...")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, verbose=False)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        print("Training complete!")


def collect_human_preference(world, traj1, traj2, display=True):
    """
    Collect a human preference between two trajectories.
    
    @param world: SciWrld instance
    @param traj1: first trajectory (list of positions)
    @param traj2: second trajectory (list of positions)
    @param display: whether to display trajectories visually
    @return: preferred trajectory index (0 or 1)
    """
    if display:
        print("\n" + "="*50)
        print("TRAJECTORY COMPARISON")
        print("="*50)
        
        # This would use world.sample_trajectories() visualization
        # For now, simple text prompt
        print(f"\nTrajectory 1: {len(traj1)} steps")
        print(f"Trajectory 2: {len(traj2)} steps")
    
    while True:
        try:
            choice = int(input("\nWhich trajectory is better? (1 or 2): "))
            if choice in [1, 2]:
                return choice - 1  # Convert to 0-indexed
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number (1 or 2)")
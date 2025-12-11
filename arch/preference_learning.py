"""
Preference Learning for RLHF.

Provides:
- PreferenceDataset: Stores trajectory pairs with human preferences
- PreferenceLearner: Trains reward model from preferences using Bradley-Terry model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class PreferenceDataset(Dataset):
    """
    Dataset of human preferences between trajectory pairs.
    
    Each sample: (traj1_states, traj2_states, preference)
    where preference is 0 if traj1 preferred, 1 if traj2 preferred.
    """
    
    def __init__(self):
        self.traj1_list = []
        self.traj2_list = []
        self.preferences = []
    
    def add_preference(self, traj1_states, traj2_states, preferred_idx):
        """
        Add a preference comparison.
        
        @param traj1_states: encoded states for trajectory 1, shape (T, state_dim)
        @param traj2_states: encoded states for trajectory 2, shape (T, state_dim)
        @param preferred_idx: 0 if traj1 preferred, 1 if traj2 preferred
        """
        # Convert to numpy arrays
        traj1 = np.array(traj1_states, dtype=np.float32)
        traj2 = np.array(traj2_states, dtype=np.float32)
        
        self.traj1_list.append(traj1)
        self.traj2_list.append(traj2)
        self.preferences.append(preferred_idx)
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.traj1_list[idx], dtype=torch.float32),
            torch.tensor(self.traj2_list[idx], dtype=torch.float32),
            torch.tensor(self.preferences[idx], dtype=torch.long)
        )
    
    def get_statistics(self):
        """Return dataset statistics."""
        if len(self) == 0:
            return {"count": 0}
        
        pref_counts = np.bincount(self.preferences, minlength=2)
        return {
            "count": len(self),
            "traj1_preferred": int(pref_counts[0]),
            "traj2_preferred": int(pref_counts[1]),
            "avg_traj_length": np.mean([len(t) for t in self.traj1_list])
        }


class PreferenceLearner:
    """
    Trains a reward model from human preferences.
    
    Uses Bradley-Terry model:
    P(traj1 > traj2) = exp(R(traj1)) / (exp(R(traj1)) + exp(R(traj2)))
                     = σ(R(traj1) - R(traj2))
    
    Loss is cross-entropy between predicted and actual preferences.
    """
    
    def __init__(self, reward_model, learning_rate=1e-3):
        """
        @param reward_model: NeuralRewardModel or NeuralRewardEnsemble
        @param learning_rate: optimizer learning rate
        """
        self.reward_model = reward_model
        self.optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def compute_preference_loss(self, traj1_states, traj2_states, preferences):
        """
        Compute Bradley-Terry cross-entropy loss.
        
        @param traj1_states: (batch, T, state_dim) tensor
        @param traj2_states: (batch, T, state_dim) tensor
        @param preferences: (batch,) tensor of preferred indices
        @return: scalar loss
        """
        batch_size = traj1_states.shape[0]
        
        # Compute trajectory rewards
        r1_list = []
        r2_list = []
        
        for i in range(batch_size):
            r1 = self.reward_model.predict_trajectory_reward(traj1_states[i])
            r2 = self.reward_model.predict_trajectory_reward(traj2_states[i])
            r1_list.append(r1)
            r2_list.append(r2)
        
        r1 = torch.stack(r1_list).squeeze()  # (batch,)
        r2 = torch.stack(r2_list).squeeze()  # (batch,)
        
        # Bradley-Terry: logits = [R(traj1), R(traj2)]
        logits = torch.stack([r1, r2], dim=1)  # (batch, 2)
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits, preferences)
        
        return loss
    
    def train_step(self, traj1_states, traj2_states, preferences):
        """Single training step."""
        self.optimizer.zero_grad()
        loss = self.compute_preference_loss(traj1_states, traj2_states, preferences)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.reward_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for traj1, traj2, pref in dataloader:
            loss = self.train_step(traj1, traj2, pref)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, dataset, epochs=50, batch_size=16, verbose=True):
        """
        Train reward model from preference dataset.
        
        @param dataset: PreferenceDataset instance
        @param epochs: number of training epochs
        @param batch_size: mini-batch size
        @param verbose: print progress
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset, skipping training")
            return
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True
        )
        
        if verbose:
            print(f"Training on {len(dataset)} preferences for {epochs} epochs...")
            print(f"Dataset stats: {dataset.get_statistics()}")
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if verbose:
            print(f"Training complete! Final loss: {self.loss_history[-1]:.4f}")
    
    def evaluate_accuracy(self, dataset):
        """
        Evaluate ranking accuracy on dataset.
        
        @param dataset: PreferenceDataset
        @return: accuracy (0 to 1)
        """
        self.reward_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(len(dataset)):
                traj1, traj2, pref = dataset[i]
                
                r1 = self.reward_model.predict_trajectory_reward(traj1).item()
                r2 = self.reward_model.predict_trajectory_reward(traj2).item()
                
                predicted = 0 if r1 > r2 else 1
                
                if predicted == pref.item():
                    correct += 1
                total += 1
        
        return correct / max(total, 1)


class EnsemblePreferenceLearner:
    """
    Trains an ensemble of reward models from preferences.
    
    Each ensemble member is trained on a bootstrap sample
    of the preference dataset.
    """
    
    def __init__(self, ensemble, learning_rate=1e-3):
        """
        @param ensemble: NeuralRewardEnsemble
        @param learning_rate: optimizer learning rate
        """
        self.ensemble = ensemble
        
        # Separate optimizer for each model
        self.optimizers = [
            optim.Adam(model.parameters(), lr=learning_rate)
            for model in ensemble.models
        ]
        
        self.loss_history = []
    
    def train(self, dataset, epochs=50, batch_size=16, verbose=True):
        """
        Train ensemble with bootstrap sampling.
        
        Each model sees a different bootstrap sample of the data.
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset")
            return
        
        if verbose:
            print(f"Training ensemble of {self.ensemble.num_models} models...")
        
        for model_idx, (model, optimizer) in enumerate(
            zip(self.ensemble.models, self.optimizers)
        ):
            if verbose:
                print(f"\n  Training model {model_idx + 1}/{self.ensemble.num_models}")
            
            # Bootstrap sample
            indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
            
            # Create learner for this model
            learner = PreferenceLearner(model, learning_rate=0)  # We'll use our optimizer
            learner.optimizer = optimizer
            
            # Create subset dataset
            subset = BootstrapSubset(dataset, indices)
            
            learner.train(subset, epochs=epochs, batch_size=batch_size, verbose=False)
            
            if verbose:
                acc = learner.evaluate_accuracy(dataset)
                print(f"    Final accuracy: {acc:.2%}")


class BootstrapSubset(Dataset):    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
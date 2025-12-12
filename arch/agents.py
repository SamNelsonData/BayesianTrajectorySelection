"""
Agent classes for SciWrld environment.

Provides:
- Agent: Base agent with random movement
- PolicyAgent: Agent controlled by neural network policy
"""
import torch

import numpy as np
from numpy.random import choice, seed as np_seed
from copy import deepcopy


class Agent:

    # Action mapping: 0=Up, 1=Left, 2=Down, 3=Right
    ACTIONS = {
        0: (-1, 0),   # Up
        1: (0, -1),   # Left
        2: (1, 0),    # Down
        3: (0, 1),    # Right
    }
    
    def __init__(self, position, world, **kwargs):

        self.position = list(position)
        self.world = world
        self.battery = 2
        
        # Precompute valid transitions for each cell
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        rows, cols = self.world.size
        
        # For each cell, store list of valid action indices
        self.valid_actions = {}
        
        for r in range(rows):
            for c in range(cols):
                valid = []
                for action, (dr, dc) in self.ACTIONS.items():
                    nr, nc = r + dr, c + dc
                    if self.world.is_valid_position((nr, nc)):
                        valid.append(action)
                self.valid_actions[(r, c)] = valid
    
    def get_valid_actions(self):
        pos = tuple(self.position)
        return self.valid_actions.get(pos, [])
    
    def act(self, action=None):

        valid = self.get_valid_actions()
        
        if not valid:
            return tuple(self.position)
        
        if action is None:
            action = choice(valid)
        elif action not in valid:
            # Invalid action: stay in place or pick random valid
            return tuple(self.position)
        
        dr, dc = self.ACTIONS[action]
        self.position[0] += dr
        self.position[1] += dc
        
        return tuple(self.position)
    
    def generate_trajectory(self, steps=8, seed=None, policy_fn=None):

        original_pos = self.position.copy()
        original_battery = self.battery
        
        if seed is not None:
            np_seed(seed)
        
        trajectory = [tuple(self.position)]
        
        for _ in range(steps):
            if policy_fn is not None:
                action = policy_fn(self)
            else:
                action = None  # Random action
            
            new_pos = self.act(action)
            trajectory.append(new_pos)
        
        # Restore original state
        self.position = original_pos
        self.battery = original_battery
        
        return trajectory
    
    def __repr__(self):
        return f"Agent(pos={self.position}, battery={self.battery})"


class PolicyAgent(Agent):
    
    def __init__(self, position, world, policy_network=None, **kwargs):

        super().__init__(position, world, **kwargs)
        self.policy = policy_network
    
    def get_state_features(self):

        from arch.sciwrld import encode_state
        return encode_state(self.world, self.position)
    
    def act(self, action=None, deterministic=False):
 
        if action is not None:
            return super().act(action)
        
        if self.policy is None:
            return super().act(None)  # Random action
                
        # Get state features
        state = self.get_state_features()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        # Get action from policy
        action_tensor, _ = self.policy.get_action(state_tensor, deterministic=deterministic)
        action = int(action_tensor.item())
        
        # Execute action (with validity check in parent)
        valid = self.get_valid_actions()
        if action not in valid:
            # Policy chose invalid action: stay in place
            return tuple(self.position)
        
        return super().act(action)
    
    def generate_trajectory(self, steps=8, seed=None, policy_fn=None, deterministic=False):

        if policy_fn is None and self.policy is not None:
            # Use the policy network
            def policy_fn(agent):
                state = agent.get_state_features()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _ = agent.policy.get_action(state_tensor, deterministic=deterministic)
                return int(action.item())
        
        return super().generate_trajectory(steps=steps, seed=seed, policy_fn=policy_fn)


def trajectory_to_string(trajectory):
    if len(trajectory) < 2:
        return ""
    
    arrows = {
        (-1, 0): '↑', (1, 0): '↓',
        (0, -1): '←', (0, 1): '→',
    }
    
    result = []
    for i in range(len(trajectory) - 1):
        curr = trajectory[i]
        next_ = trajectory[i + 1]
        delta = (next_[0] - curr[0], next_[1] - curr[1])
        result.append(arrows.get(delta, '•'))
    
    return ''.join(result)

def trajectory_similarity(traj1, traj2):

    set1 = set(traj1)
    set2 = set(traj2)
    overlap = len(set1 & set2)
    total = max(len(set1), len(set2))
    return overlap / total if total > 0 else 1.0
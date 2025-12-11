"""
Agent classes for SciWrld environment.

Provides:
- Agent: Base agent with random movement
- PolicyAgent: Agent controlled by neural network policy
"""

import numpy as np
from numpy.random import choice, seed as np_seed
from copy import deepcopy


class Agent:
    """
    Base agent that moves randomly in the grid.
    Used for initial trajectory generation before learning.
    """
    
    # Action mapping: 0=Up, 1=Left, 2=Down, 3=Right
    ACTIONS = {
        0: (-1, 0),   # Up
        1: (0, -1),   # Left
        2: (1, 0),    # Down
        3: (0, 1),    # Right
    }
    
    def __init__(self, position, world, **kwargs):
        """
        @param position: starting [row, col] position
        @param world: reference to SciWrld instance
        """
        self.position = list(position)
        self.world = world
        self.battery = 2
        
        # Precompute valid transitions for each cell
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """Build matrix of valid moves from each position."""
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
        """Return list of valid action indices from current position."""
        pos = tuple(self.position)
        return self.valid_actions.get(pos, [])
    
    def act(self, action=None):
        """
        Take an action and update position.
        
        @param action: action index (0-3), or None for random
        @return: new position tuple
        """
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
        """
        Generate a trajectory of given length.
        
        IMPORTANT: This modifies agent position temporarily, then restores it.
        Does NOT modify world state.
        
        @param steps: number of steps in trajectory
        @param seed: optional random seed
        @param policy_fn: optional function(agent) -> action index
        @return: list of positions [(r, c), ...]
        """
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
    """
    Agent controlled by a neural network policy.
    Used after learning a reward function.
    """
    
    def __init__(self, position, world, policy_network=None, **kwargs):
        """
        @param position: starting [row, col]
        @param world: SciWrld instance
        @param policy_network: PolicyNetwork instance
        """
        super().__init__(position, world, **kwargs)
        self.policy = policy_network
    
    def get_state_features(self):
        """
        Get state features for policy network input.
        Uses the same encoding as the reward model.
        """
        from arch.sciwrld import encode_state
        return encode_state(self.world, self.position)
    
    def act(self, action=None, deterministic=False):
        """
        Take action using policy network.
        
        @param action: if provided, use this action (ignores policy)
        @param deterministic: if True, take argmax action
        @return: new position tuple
        """
        if action is not None:
            return super().act(action)
        
        if self.policy is None:
            return super().act(None)  # Random action
        
        import torch
        
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
        """
        Generate trajectory using policy network.
        
        @param steps: trajectory length
        @param seed: random seed
        @param policy_fn: override policy (if None, use self.policy)
        @param deterministic: use deterministic policy
        """
        if policy_fn is None and self.policy is not None:
            # Use the policy network
            def policy_fn(agent):
                import torch
                state = agent.get_state_features()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _ = agent.policy.get_action(state_tensor, deterministic=deterministic)
                return int(action.item())
        
        return super().generate_trajectory(steps=steps, seed=seed, policy_fn=policy_fn)


def trajectory_to_string(trajectory):
    """Convert trajectory to arrow string for display."""
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
    """
    Compute overlap between two trajectories.
    
    @return: fraction of positions that overlap (0 to 1)
    """
    set1 = set(traj1)
    set2 = set(traj2)
    overlap = len(set1 & set2)
    total = max(len(set1), len(set2))
    return overlap / total if total > 0 else 1.0